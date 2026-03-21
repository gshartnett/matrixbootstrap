from matrixbootstrap.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
)


def test_instantiate_single_trace_operator():
    """
    Make sure that the single trace operator instantiation is
    insensitive to whether the input is a tuple or a string
    for the special case of a degree 1 term.
    """
    op1 = MatrixOperator(data={"P2": 0.5})
    op2 = MatrixOperator(data={("P2",): 0.5})
    assert op1 == op2


def test_single_trace_commutator_onematrix():
    """
    Test the Hamiltonian constraint <[H,O]> = 0 for the case
    of O = tr(XP) and H given  by the single-matrix QM model
    studied in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.041601.
    This is used to derive the constraint Eq. (8)
    """
    matrix_system = MatrixSystem(
        operator_basis=["X", "P"],
        commutation_rules_concise={
            ("P", "X"): -1j,
        },
        hermitian_dict={"P": True, "X": True},
    )
    OP1 = SingleTraceOperator(data={("X", "P"): 1})
    OP2 = SingleTraceOperator(
        data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): 7}
    )
    assert matrix_system.single_trace_commutator(OP1, OP2) == SingleTraceOperator(
        data={("P", "P"): 2j, ("X", "X"): -2j, ("X", "X", "X", "X"): -4 * 7j}
    )
    assert matrix_system.single_trace_commutator2(OP1, OP2) == SingleTraceOperator(
        data={("P", "P"): 2j, ("X", "X"): -2j, ("X", "X", "X", "X"): -4 * 7j}
    )


def test_single_trace_commutator_twomatrix():
    """
    Test the Hamiltonian constraint <[H,O]> = 0 for an example in Han et al,
    see: https://github.com/hanxzh94/matrix-bootstrap/blob/b16d407ae2c7436e2f17a00fb92427d10b94012d/trace.py#L204
    """
    g, h = 1, 3
    matrix_system = MatrixSystem(
        operator_basis=["P1", "X1", "P2", "X2"],
        commutation_rules_concise={
            ("P1", "X1"): -1j,
            ("P2", "X2"): -1j,
        },
        hermitian_dict={"P1": True, "X1": True, "P2": True, "X2": True},
    )
    hamiltonian = SingleTraceOperator(
        data={
            ("P1", "P1"): 1,
            ("X1", "X1"): 1,
            ("P2", "P2"): 1,
            ("X2", "X2"): 1,
            ("X1", "X2", "X1", "X2"): -2 * g,
            ("X1", "X1", "X2", "X2"): 2 * g + 2 * h,
            ("X1", "X1", "X1", "X1"): h,
            ("X2", "X2", "X2", "X2"): h,
            # ("X1", "X1", "X2", "X2"): 2*h,
        }
    )

    S2 = SingleTraceOperator(data={("X1", "P2"): 1, ("X2", "P1"): -1})

    commutator = SingleTraceOperator(
        data={
            ("P1", "P2"): -2j,
            ("X2", "X1"): -2j,
            ("P2", "P1"): 2j,
            ("X1", "X2"): 2j,
            ("X1", "X1", "X2", "X1"): -4j,
            ("X2", "X2", "X1", "X2"): 4j,
            ("X1", "X2", "X1", "X1"): 8j,
            ("X1", "X1", "X1", "X2"): 8j,
            ("X2", "X1", "X2", "X2"): -8j,
            ("X2", "X2", "X2", "X1"): -8j,
            ("X2", "X1", "X1", "X1"): -12j,
            ("X1", "X2", "X2", "X2"): 12j,
        }
    )

    assert matrix_system.single_trace_commutator(hamiltonian, S2) == commutator


# assert matrix_system.single_trace_commutator2(S2, hamiltonian) == commutator


def test_zero_single_trace_operator():
    """
    Test edge cases involving the zero operator
    """
    zero = SingleTraceOperator(data={(): 0})

    assert len(zero) == 0

    # 0 * <tr(O)>
    assert SingleTraceOperator(data={("P", "P"): 0}) == zero


def test_zero_double_trace_operator():
    """
    Test edge cases involving the zero operator
    """
    zero = SingleTraceOperator(data={(): 0})
    assert zero * zero == zero


def test_single_trace_component_of_double_trace():
    op = SingleTraceOperator(
        data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): 7}
    )
    one = SingleTraceOperator(data={(): 1})
    assert (one * op).get_single_trace_component() == op
    assert (op * one).get_single_trace_component() == op


def test_get_real_and_imag_parts():
    st_op = SingleTraceOperator(data={"A": 1, "B": 2, "C": 3})
    assert st_op.is_real()

    st_op = SingleTraceOperator(data={"A": 1, "B": 2, "C": 3j})
    assert not st_op.is_real()

    st_op = SingleTraceOperator(data={"A": 1j, "B": 2j, "C": 3j})
    assert st_op.is_imag()

    st_op = SingleTraceOperator(data={"A": 1j, "B": 2j, "C": 3})
    assert not st_op.is_imag()
    assert 1j * st_op.get_imag_part() == SingleTraceOperator(data={"A": 1j, "B": 2j})
    assert st_op.get_real_part() == SingleTraceOperator(data={"C": 3})
