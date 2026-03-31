"""
Validate L=2 invariant basis gives same energy bound as non-invariant + symmetry.

Expected: both should give roughly the same lower bound on <H> for nu=1.5, lambda=1.
"""

import logging
import time

from matrixbootstrap.bootstrap_complex import BootstrapSystemComplex
from matrixbootstrap.models import MiniBMN
from matrixbootstrap.solver_newton import solve_bootstrap

logging.basicConfig(level=logging.INFO)

model = MiniBMN(couplings={"nu": 1.5, "lambda": 1.0})

SOLVE_KWARGS = dict(
    maxiters=20,
    maxiters_cvxpy=500,
    init_scale=1e-2,
    reg=1e-3,
    eps_abs=1e-5,
    eps_rel=1e-5,
    tol=1e-7,
    radius=1e10,
    cvxpy_solver="SCS",
)

print("\n=== Non-invariant L=2 (with SO(3) symmetry) ===")
t0 = time.time()
bs_noninv = BootstrapSystemComplex(
    matrix_system=model.matrix_system,
    hamiltonian=model.hamiltonian,
    gauge_generator=model.gauge_generator,
    symmetry_generators=model.symmetry_generators,
    max_degree_L=2,
    odd_degree_vanish=True,
    simplify_quadratic=True,
    use_invariant_basis=False,
)
bs_noninv.build_null_space_matrix()
bs_noninv.build_quadratic_constraints()
bs_noninv.build_bootstrap_table()
print(
    f"Built in {time.time()-t0:.1f}s, param_dim={bs_noninv.param_dim_null}, bootstrap_dim={bs_noninv.bootstrap_matrix_dim}"
)

param, result = solve_bootstrap(
    bs_noninv,
    st_operator_to_minimize=model.hamiltonian,
    **SOLVE_KWARGS,
)
e_noninv = result["prob.value"] if result else float("nan")
print(f"Energy bound (non-invariant): {e_noninv:.6f}")

print("\n=== Invariant basis L=2 (multi-block PSD) ===")
t0 = time.time()
bs_inv = BootstrapSystemComplex(
    matrix_system=model.matrix_system,
    hamiltonian=model.hamiltonian,
    gauge_generator=model.gauge_generator,
    symmetry_generators=model.symmetry_generators,
    max_degree_L=2,
    odd_degree_vanish=True,
    simplify_quadratic=True,
    use_invariant_basis=True,
)
bs_inv.build_null_space_matrix()
bs_inv.build_quadratic_constraints()
bs_inv.build_bootstrap_table()
print(
    f"Built in {time.time()-t0:.1f}s, param_dim={bs_inv.param_dim_null}, bootstrap_dim={bs_inv.bootstrap_matrix_dim}"
)
if hasattr(bs_inv, "extra_bootstrap_tables") and bs_inv.extra_bootstrap_tables:
    for q, t in bs_inv.extra_bootstrap_tables.items():
        n_q = int(t.shape[0] ** 0.5)
        print(f"  extra block q={q}: {n_q}x{n_q}")

param_inv, result_inv = solve_bootstrap(
    bs_inv,
    st_operator_to_minimize=bs_inv.hamiltonian,  # eigenbasis hamiltonian
    **SOLVE_KWARGS,
)
e_inv = result_inv["prob.value"] if result_inv else float("nan")
print(f"Energy bound (invariant):     {e_inv:.6f}")

print(f"\nDifference: {abs(e_inv - e_noninv):.4e}")
