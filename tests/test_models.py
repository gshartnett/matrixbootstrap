import pytest

from matrixbootstrap.algebra import SingleTraceOperator
from matrixbootstrap.models import MiniBFSS, MiniBMN, OneMatrix, TwoMatrix


def cyclic_reduce(op: SingleTraceOperator) -> SingleTraceOperator:
    """
    Collapse a SingleTraceOperator by summing coefficients of cyclic equivalents.

    Inside a trace, tr(A1 A2 ... An) = tr(A2 ... An A1), so many algebraically
    distinct strings represent the same expectation value. This function maps
    each string to its lexicographically minimal cyclic rotation, then sums
    coefficients that share the same canonical form.
    """
    canonical: dict = {}
    for tup, coeff in op:
        if len(tup) == 0:
            canon = tup
        else:
            canon = min(tup[i:] + tup[:i] for i in range(len(tup)))
        canonical[canon] = canonical.get(canon, 0) + coeff
    return SingleTraceOperator(data=canonical)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    OneMatrix(couplings={"g2": 1.0, "g4": 1.0, "g6": 0.0}),
    TwoMatrix(couplings={"g2": 1.0, "g4": 1.0}),
    MiniBFSS(couplings={"lambda": 1.0}),
    MiniBMN(couplings={"nu": 1.0, "lambda": 1.0}),
], ids=["OneMatrix", "TwoMatrix", "MiniBFSS", "MiniBMN"])
def model(request):
    return request.param


@pytest.fixture(params=[
    TwoMatrix(couplings={"g2": 1.0, "g4": 1.0}),
    MiniBFSS(couplings={"lambda": 1.0}),
    MiniBMN(couplings={"nu": 1.0, "lambda": 1.0}),
], ids=["TwoMatrix", "MiniBFSS", "MiniBMN"])
def model_with_symmetry(request):
    return request.param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hamiltonian_is_hermitian(model):
    """
    The Hamiltonian must satisfy H† = H for every model.

    This checks that the Hamiltonian as defined in the algebra (with the chosen
    conventions for X, Pi) is self-adjoint, which is required for the eigenvalues
    to be real physical energies.
    """
    H = model.hamiltonian
    H_dag = model.matrix_system.hermitian_conjugate(H)
    assert (H - H_dag).is_zero()


def test_symmetry_generators_commute_with_hamiltonian(model_with_symmetry):
    """
    Every symmetry generator M must satisfy [H, M] = 0 up to cyclic trace identities.

    single_trace_commutator returns an algebraic expression that may contain pairs
    like tr(AB) - tr(BA) which are identical operators (tr(AB) = tr(BA) inside a
    trace) but represented by different strings. cyclic_reduce collapses these
    before checking for zero, giving the correct operator-level statement.

    This is a necessary condition for M to generate a true symmetry: it implies
    that the charge is conserved and that the Hilbert space decomposes into
    invariant sectors under the symmetry group.
    """
    for sym_gen in model_with_symmetry.symmetry_generators:
        commutator = model_with_symmetry.matrix_system.single_trace_commutator(
            model_with_symmetry.hamiltonian, sym_gen
        )
        assert cyclic_reduce(commutator).is_zero()
