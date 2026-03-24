"""
Bootstrap vs Born-Oppenheimer comparison for the d=2 two-matrix model.

Reproduces HKK Fig 3a: energy vs g^2*N ('t Hooft coupling) at m=1, L=3.

Convention mapping (HKK <-> ours):
    m          = 1 (fixed)      <->   g2 = 1.0
    lambda_HKK = g_HKK^2       <->   g4 = 2 * lambda_HKK  (since g4/4 = g^2/2)
    E_HKK      = 2 * E_ours    (the /2 factor in BornOppenheimer.solve with g2/g4)

X-axis in HKK Fig 3a: lambda = g^2 * N (the 't Hooft coupling)
Y-axis in HKK Fig 3a: E (energy in HKK conventions) = 2 * E_ours

BO upper/lower bounds: both are variational upper bounds on the true BO minimum.
The band arises from grid resolution — a coarser/narrower grid is more restrictive
and gives higher energy (BO "upper"); a finer/wider grid gives lower energy (BO "lower").

Usage:
    python scripts/two_matrix_bo_comparison.py           # generate configs + run + plot
    python scripts/two_matrix_bo_comparison.py --plot    # plot only (from saved results)
"""

import json
import logging
import os

import fire
import matplotlib.pyplot as plt
import numpy as np

from matrixbootstrap.born_oppenheimer import BornOppenheimer
from matrixbootstrap.config_utils import (
    generate_config_two_matrix,
    run_all_configs,
)

logging.basicConfig(level=logging.INFO)

# ── parameters ────────────────────────────────────────────────────────────────
L = 3
G2 = 1.0  # m=1 fixed (HKK convention)

# FAST=True: 6 lambda points (for quick iteration)
# FAST=False: 13 lambda points (for final figure)
FAST = True

if FAST:
    LAMBDA_VALUES = np.round(np.linspace(0.25, 8.0, 6), decimals=4).tolist()
    MAXITERS = 30
else:
    LAMBDA_VALUES = np.round(np.linspace(0.0, 8.0, 13), decimals=4).tolist()
    MAXITERS = 100

# BO grids: lower bound uses E_BO (O(n^2)), upper bound uses E_var (O(n^3))
BO_X_GRID_LOWER = np.linspace(-12, 12, 200)  # fine grid for lower bound
BO_X_GRID_UPPER = np.linspace(-12, 12, 80)  # coarser grid for O(n^3) upper bound

REG_VALUES = [1e-3, 1e-4, 1e-5]
HKK_CSV = "HKK_extracted_data_Fig3a.csv"

CONFIG_DIR = f"TwoMatrix_BO_comparison_L{L}_g2_{G2}"


# ── config generation ─────────────────────────────────────────────────────────
def generate_configs():
    for lam in LAMBDA_VALUES:
        g4 = 2.0 * lam  # g4/4 = g^2/2  =>  g4 = 2*g^2 = 2*lambda
        for reg in REG_VALUES:
            generate_config_two_matrix(
                config_dir=CONFIG_DIR,
                g2=G2,
                g4=g4,
                st_operator_to_minimize="energy",
                st_operators_evs_to_set=None,
                max_degree_L=L,
                impose_symmetries=True,
                load_from_previously_computed=True,
                optimization_method="newton",
                cvxpy_solver="CLARABEL",
                maxiters=MAXITERS,
                maxiters_cvxpy=500,
                init_scale=1e-2,
                reg=reg,
                eps_abs=1e-5,
                eps_rel=1e-5,
                tol=1e-7,
                radius=1e5,
                use_factorization_block=True,
            )


# ── plotting ──────────────────────────────────────────────────────────────────
def make_plot():
    import yaml

    results_dir = f"runs/{CONFIG_DIR}/results"
    configs_dir = f"runs/{CONFIG_DIR}/configs"

    if not os.path.exists(results_dir):
        print(f"No results directory found: {results_dir}")
        return

    # load bootstrap results, indexed by (lambda, reg)
    # filter out bad results: optimizer escaped to radius boundary (||x|| >> 1)
    # or quadratic constraints are badly violated
    MAX_QUAD_VIOLATION = 1.0
    data = {}
    n_bad = 0
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            result = json.load(f)
        with open(os.path.join(configs_dir, fname.replace(".json", ".yaml"))) as f:
            config = yaml.safe_load(f)
        g4 = config["model"]["couplings"]["g4"]
        reg = config["optimizer"]["reg"]
        lam = float(np.round(g4 / 2, decimals=4))  # lambda = g4/2
        energy = result.get("energy")
        quad_viol = result.get("max_quad_constraint_violation", 0.0)
        if energy is None:
            continue
        if quad_viol > MAX_QUAD_VIOLATION:
            print(
                f"  Dropping (lam={lam}, reg={reg:.0e}): quad_viol={quad_viol:.2e}, E={energy:.3f}"
            )
            n_bad += 1
            continue
        data[(lam, reg)] = energy

    if not data:
        print(f"No results found in {results_dir}")
        return
    print(
        f"Loaded {len(data)} valid bootstrap results ({n_bad} dropped as unconverged)."
    )

    # our BO band — computed over all LAMBDA_VALUES (independent of bootstrap completion)
    # lower: minimize E_BO (O(n^2)); upper: minimize E_var (O(n^3), use coarser grid)
    bo_lambdas = sorted(LAMBDA_VALUES)
    print(f"Computing BO bounds at {len(bo_lambdas)} lambda values...")
    bo_upper, bo_lower = [], []
    for lam in bo_lambdas:
        g4 = 2.0 * lam
        bo = BornOppenheimer(g2=G2, g4=g4)
        bo_lower.append(bo.solve(BO_X_GRID_LOWER).fun)
        print(f"  lambda={lam:.4f}: lower done, computing upper (O(n^3))...")
        bo_upper.append(bo.solve_upper(BO_X_GRID_UPPER).fun)

    # HKK extracted data — x is lambda=g^2*N, y is E_HKK -> E_ours = y/2
    hkk = {}
    if os.path.exists(HKK_CSV):
        raw = np.genfromtxt(
            HKK_CSV, delimiter=",", skip_header=2, filling_values=np.nan
        )

        # columns: L3_x, L3_y, L4_x, L4_y, BO_upper_x, BO_upper_y, BO_lower_x, BO_lower_y
        def clean(x, y):
            mask = ~(np.isnan(x) | np.isnan(y))
            return x[mask], y[mask] / 2  # x = lambda, y: E_HKK -> E_ours

        hkk["L3_x"], hkk["L3_y"] = clean(raw[:, 0], raw[:, 1])
        hkk["L4_x"], hkk["L4_y"] = clean(raw[:, 2], raw[:, 3])
        hkk["bo_up_x"], hkk["bo_up_y"] = clean(raw[:, 4], raw[:, 5])
        hkk["bo_lo_x"], hkk["bo_lo_y"] = clean(raw[:, 6], raw[:, 7])
        print(f"Loaded HKK data from {HKK_CSV}")
    else:
        print(f"HKK CSV not found at {HKK_CSV}, skipping reference data.")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    COLOR_HKK = "0.35"  # gray for HKK reference
    COLOR_OURS = "#1f77b4"  # blue for our results

    # HKK BO band (gray: dashed=upper, solid=lower, shaded between)
    if False:
        # interpolate upper onto lower's x-grid for fill_between
        up_interp = np.interp(hkk["bo_lo_x"], hkk["bo_up_x"], hkk["bo_up_y"])
        ax.fill_between(
            hkk["bo_lo_x"], up_interp, hkk["bo_lo_y"], alpha=0.15, color=COLOR_HKK
        )
        ax.plot(
            hkk["bo_up_x"],
            hkk["bo_up_y"],
            "--",
            color=COLOR_HKK,
            lw=1.5,
            label="HKK BO upper",
        )
        ax.plot(
            hkk["bo_lo_x"],
            hkk["bo_lo_y"],
            "-",
            color=COLOR_HKK,
            lw=1.5,
            label="HKK BO lower",
        )

    # our BO band (blue: dashed=upper, solid=lower, shaded between)
    ax.fill_between(bo_lambdas, bo_upper, bo_lower, alpha=0.15, color=COLOR_OURS)
    ax.plot(
        bo_lambdas,
        bo_upper,
        "--",
        color=COLOR_OURS,
        lw=1.5,
        label="Our BO upper ($E_\\mathrm{var}$)",
    )
    ax.plot(
        bo_lambdas,
        bo_lower,
        "-",
        color=COLOR_OURS,
        lw=1.5,
        label="Our BO lower ($E_\\mathrm{BO}$)",
    )

    # HKK bootstrap (gray markers)
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

    # our bootstrap (blue markers, varying alpha per reg)
    reg_alphas = np.linspace(0.45, 1.0, len(REG_VALUES))
    for alpha, reg in zip(reg_alphas, REG_VALUES):
        lams, energies = [], []
        for lam in sorted(LAMBDA_VALUES):
            e = data.get((lam, reg))
            if e is not None:
                lams.append(lam)
                energies.append(e)
        if lams:
            ax.plot(
                lams,
                energies,
                "o-",
                color=COLOR_OURS,
                alpha=alpha,
                label=f"Our bootstrap $L={L}$, reg={reg:.0e}",
                ms=5,
            )

    ax.set_xlabel(r"$\lambda = g^2 N$", fontsize=13)
    ax.set_ylabel(r"$E$  (our conventions, $= E_\mathrm{HKK}/2$)", fontsize=12)
    ax.set_title(rf"Two-matrix model, $L={L}$, $m=1$ (HKK Fig 3a)", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"runs/{CONFIG_DIR}/bo_comparison.pdf"
    os.makedirs(f"runs/{CONFIG_DIR}", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()
    plt.close()


# ── entrypoint ────────────────────────────────────────────────────────────────
def main(plot=False):
    if plot:
        make_plot()
        return
    generate_configs()
    run_all_configs(
        config_dir=CONFIG_DIR,
        parallel=True,
        max_workers=min(3, len(LAMBDA_VALUES) * len(REG_VALUES)),
        check_if_exists_already=True,
    )
    make_plot()


if __name__ == "__main__":
    fire.Fire(main)
