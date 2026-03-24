import numpy as np
import pytest

from matrixbootstrap.bootstrap import BootstrapSystem
from matrixbootstrap.bootstrap_complex import BootstrapSystemComplex
from matrixbootstrap.models import OneMatrix


@pytest.fixture(scope="module")
def one_matrix_bootstrap_L1():
    """
    A minimal BootstrapSystem for the harmonic oscillator (g4=g6=0) at L=1.

    The L=1 bootstrap tracks all single-trace monomials of degree <= 2L=2 in X
    and Pi: {1, X, Pi, X², X·Pi, Pi·X, Pi²} — 7 operators total. The bootstrap
    matrix is 3x3, indexed by the degree-<=1 monomials {1, X, Pi}. With
    odd_degree_vanish=True, <X> = <Pi> = 0, leaving 5 free expectation values.
    Cheap to build; used as a shared fixture across tests.
    """
    model = OneMatrix(couplings={"g2": 1.0, "g4": 0.0, "g6": 0.0})
    return BootstrapSystem(
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=1,
        odd_degree_vanish=True,
        simplify_quadratic=False,
        structural_cache_path=None,
        config_cache_path=None,
    )


@pytest.fixture(scope="module")
def one_matrix_real_and_complex_L2():
    """
    BootstrapSystem and BootstrapSystemComplex built from the same OneMatrix model at L=2.

    Used to verify that the complex bootstrap produces a Hermitian bootstrap matrix and
    that its SDP lower bound on the ground-state energy is consistent with the real system.
    simplify_quadratic=False keeps build time short for tests.
    """
    model = OneMatrix(couplings={"g2": 1.0, "g4": 0.0, "g6": 0.0})
    real_sys = BootstrapSystem(
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=2,
        odd_degree_vanish=True,
        simplify_quadratic=False,
        checkpoint_path=None,
    )
    complex_sys = BootstrapSystemComplex(
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=2,
        odd_degree_vanish=True,
        simplify_quadratic=False,
        checkpoint_path=None,
    )
    return real_sys, complex_sys


def test_complex_bootstrap_matrix_is_hermitian(one_matrix_real_and_complex_L2):
    """
    The bootstrap matrix produced by BootstrapSystemComplex must be Hermitian.

    For random null-space parameter vectors, M[i,j] = M[j,i]* must hold to
    machine precision. This catches the sign error in the reality constraints
    that previously made M anti-symmetric in some entries.
    """
    _, complex_sys = one_matrix_real_and_complex_L2
    complex_sys.build_null_space_matrix()
    complex_sys.build_bootstrap_table()

    rng = np.random.default_rng(42)
    for _ in range(5):
        param = rng.standard_normal(complex_sys.param_dim_null)
        mat = complex_sys.get_bootstrap_matrix(param)
        assert np.allclose(
            mat, mat.conj().T, atol=1e-12
        ), "Bootstrap matrix is not Hermitian for a random parameter vector"


def test_complex_bootstrap_table_is_complex_dtype(one_matrix_real_and_complex_L2):
    """
    The bootstrap table for BootstrapSystemComplex must have complex dtype.

    The bootstrap matrix M[i,j] = sign*(vR[p] + i*vI[p]) is generically complex,
    so the table that accumulates its entries must use complex128. Previously the
    table used float64, silently discarding all imaginary contributions.
    """
    _, complex_sys = one_matrix_real_and_complex_L2
    complex_sys.build_null_space_matrix()
    complex_sys.build_bootstrap_table()
    assert np.iscomplexobj(
        complex_sys.bootstrap_table_sparse.toarray()
    ), "bootstrap_table_sparse should have complex dtype for BootstrapSystemComplex"


def test_null_space_satisfies_linear_constraints(one_matrix_bootstrap_L1):
    """
    Every vector in the null space of the linear constraints must satisfy
    those constraints exactly: L K = 0.

    The null space K parametrises all physically admissible operator expectation
    value vectors — those consistent with gauge invariance, Hermiticity, and the
    equations of motion <[H, O]> = 0. Verifying L K = 0 confirms that the constraint
    construction and null space computation are mutually consistent.
    """
    bootstrap = one_matrix_bootstrap_L1
    L = bootstrap.build_linear_constraints().tocsr()
    bootstrap.build_null_space_matrix()
    K = bootstrap.null_space_matrix

    residual = (L @ K).toarray()
    assert np.allclose(residual, 0, atol=1e-10)
