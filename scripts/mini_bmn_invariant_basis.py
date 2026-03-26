"""
Mini-BMN bootstrap with SO(3) invariant basis (Cartan eigenbasis).

Runs L=2 and/or L=3 and compares energy bounds.

Usage:
    python scripts/mini_bmn_invariant_basis.py           # L=2 validate + L=3 run
    python scripts/mini_bmn_invariant_basis.py --L 2     # L=2 only
    python scripts/mini_bmn_invariant_basis.py --L 3     # L=3 only
"""

import logging

import fire

from matrixbootstrap.config_utils import (
    generate_config_bmn,
    run_all_configs,
)

logging.basicConfig(level=logging.INFO)

NU_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
LAMBDA = 1.0


def run(L=2, parallel=True):
    config_dir = f"MiniBMN_invariant_L{L}"
    print(f"\n=== Mini-BMN invariant basis, L={L} ===")
    print(f"  nu values: {NU_VALUES}")
    print(f"  lambda: {LAMBDA}")

    for nu in NU_VALUES:
        generate_config_bmn(
            config_dir=config_dir,
            nu=nu,
            lambd=LAMBDA,
            optimization_method="newton",
            max_degree_L=L,
            odd_degree_vanish=True,
            simplify_quadratic=True,
            impose_symmetries=False,  # symmetry handled by invariant basis
            use_invariant_basis=True,  # <-- the new flag
            load_from_previously_computed=True,
            cvxpy_solver="SCS",
            maxiters=50,
            maxiters_cvxpy=500,
            init_scale=1e-2,
            reg=1e-3,
            eps_abs=1e-5,
            eps_rel=1e-5,
            tol=1e-7,
            radius=1e10,
            use_factorization_block=False,
        )

    run_all_configs(
        config_dir=config_dir,
        parallel=parallel,
        max_workers=min(4, len(NU_VALUES)),
        check_if_exists_already=True,
    )


if __name__ == "__main__":
    fire.Fire(run)
