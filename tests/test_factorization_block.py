"""
Tests for the augmented factorization-block constraint M_hat = [[M, v],[v^T, 1]] ⪰ 0.

This constraint is a necessary condition for large-N factorization:
    M_{IJ} = <O_I><O_J>  =>  M ⪰ v v^T  =>  M_hat ⪰ 0.

It regularizes the multi-matrix bootstrap SDP, which is otherwise ill-posed
(unbounded below) when the quadratic factorization constraints are dropped.
"""

import numpy as np
import pytest

from matrixbootstrap.algebra import SingleTraceOperator
from matrixbootstrap.bootstrap import BootstrapSystemReal
from matrixbootstrap.models import TwoMatrix


@pytest.fixture(scope="module")
def two_matrix_L2():
    """TwoMatrix BootstrapSystem at L=2 with null space and bootstrap table built."""
    model = TwoMatrix(couplings={"g2": 1.0, "g4": 2.0})
    bs = BootstrapSystemReal(
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=2,
        odd_degree_vanish=True,
        simplify_quadratic=True,
        structural_cache_path=None,
        config_cache_path=None,
    )
    bs.build_null_space_matrix()
    bs.build_bootstrap_table()
    return bs, model


# ── structural tests ──────────────────────────────────────────────────────────


def test_v_table_shape(two_matrix_L2):
    bs, _ = two_matrix_L2
    v_table = bs.build_augmented_bootstrap_table()
    assert v_table.shape == (bs.bootstrap_matrix_dim, bs.param_dim_null)


def test_v_table_is_real(two_matrix_L2):
    """v_table must be real-valued (even-degree EVs are real, odd-degree vanish)."""
    bs, _ = two_matrix_L2
    v_table = bs.build_augmented_bootstrap_table()
    assert v_table.dtype == np.float64


def test_v_table_identity_row_matches_coefficient_vector(two_matrix_L2):
    """Row for the identity operator () must equal the <1> coefficient vector."""
    bs, _ = two_matrix_L2
    v_table = bs.build_augmented_bootstrap_table()
    identity_idx = bs.bootstrap_basis_list.index(())
    expected = bs.single_trace_to_coefficient_vector(
        SingleTraceOperator(data={(): 1}), return_null_basis=True
    ).real
    assert np.allclose(v_table[identity_idx, :], expected, atol=1e-12)


def test_v_table_identity_evaluates_to_one(two_matrix_L2):
    """
    For any param satisfying <1>=1, the identity row of v_table must evaluate to 1.
    This verifies that the (n,n) corner of M_hat is correctly pinned to 1.
    """
    bs, _ = two_matrix_L2
    v_table = bs.build_augmented_bootstrap_table()

    identity_coeff = bs.single_trace_to_coefficient_vector(
        SingleTraceOperator(data={(): 1}), return_null_basis=True
    ).real
    # particular solution satisfying <1>=1
    param_particular = identity_coeff / (identity_coeff @ identity_coeff)

    # add null-space perturbation (leaves <1>=1 unchanged)
    from matrixbootstrap.linear_algebra import get_null_space_dense

    rng = np.random.default_rng(7)
    null_K = get_null_space_dense(identity_coeff.reshape(1, -1))
    param = param_particular + null_K @ rng.standard_normal(null_K.shape[1]) * 0.01

    identity_idx = bs.bootstrap_basis_list.index(())
    v_identity = v_table[identity_idx, :] @ param
    assert np.isclose(
        v_identity, 1.0, atol=1e-10
    ), f"v_0 = <1> = {v_identity:.6f}, expected 1.0"


# ── functional tests ──────────────────────────────────────────────────────────


def test_factorization_block_bounds_energy():
    """
    With use_factorization_block=True the SDP is bounded and returns a
    physically meaningful energy at small regularization.

    At g2=1, g4=2 (lambda=1) the BO lower bound is E_ours ~ 1.18.
    Without the factorization block the SDP energy diverges (>>10) at reg→0;
    with it the energy should stay within the BO bounds.

    A fresh BootstrapSystem is created here (not shared) so that solve_bootstrap
    can drive the full setup sequence, including the simplify_quadratic null-space
    rebuild, without conflicting with a pre-built bootstrap table.
    """
    from matrixbootstrap.solver_newton import solve_bootstrap

    model = TwoMatrix(couplings={"g2": 1.0, "g4": 2.0})
    bs = BootstrapSystemReal(
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=2,
        odd_degree_vanish=True,
        simplify_quadratic=True,
        structural_cache_path=None,
        config_cache_path=None,
    )

    param, result = solve_bootstrap(
        bootstrap=bs,
        st_operator_to_minimize=model.hamiltonian,
        maxiters=5,
        maxiters_cvxpy=3000,
        reg=1e-6,
        radius=1e5,
        use_factorization_block=True,
        cvxpy_solver="SCS",
    )

    assert param is not None, "Solver returned None"
    energy_coeff = bs.single_trace_to_coefficient_vector(
        model.hamiltonian, return_null_basis=True
    ).real
    energy = energy_coeff @ param
    # Energy must be finite and positive, well below the radius-driven blow-up
    assert np.isfinite(energy), f"Energy is not finite: {energy}"
    assert (
        energy < 10.0
    ), f"Energy {energy:.3f} too large — factorization block not working"
    assert energy > 0.5, f"Energy {energy:.3f} unphysically small"
