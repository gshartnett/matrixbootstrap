import numpy as np
import pytest

from matrixbootstrap.bootstrap import BootstrapSystem
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
