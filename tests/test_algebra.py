"""
Test suite for the algebra module.

This module tests the core algebraic structures used in matrix model bootstrap:
- MatrixOperator: Untraced matrix operators
- SingleTraceOperator: Single trace operators tr(...)
- DoubleTraceOperator: Double trace operators tr(...) tr(...)
- MatrixSystem: Physical systems with commutation relations

The tests cover basic functionality, arithmetic operations, error handling,
and realistic physics calculations used in BMN and BFSS models.
"""

import pytest
import numpy as np
from bmn.algebra import (
    MatrixOperator,
    MatrixSystem,
    SingleTraceOperator,
    DoubleTraceOperator,
    TOL
)


def test_instantiate_single_trace_operator():
    """
    Test operator instantiation with different input formats.

    Ensures that SingleTraceOperator creation is robust to whether
    the input uses tuple notation ("P2",) or string notation "P2"
    for degree-1 operators. This is important for user convenience
    and backward compatibility.
    """
    op1 = MatrixOperator(data={"P2": 0.5})
    op2 = MatrixOperator(data={("P2",): 0.5})
    assert op1 == op2


def test_matrix_operator_creation():
    """
    Test MatrixOperator creation with various input formats.

    Verifies:
    - Tuple key format: {("X", "P"): coeff}
    - String key format: {"X": coeff} (converted to tuple internally)
    - Tolerance filtering: coefficients below threshold are removed
    """
    # Test with tuple keys - standard format for multi-degree operators
    op1 = MatrixOperator(data={("X", "P"): 1.0, ("P", "X"): -1.0})
    assert len(op1) == 2
    assert op1.data[("X", "P")] == 1.0

    # Test with string keys - convenience format for single operators
    # Should be automatically converted to tuple format internally
    op2 = MatrixOperator(data={"X": 2.0})
    assert op2.data[("X",)] == 2.0

    # Test tolerance filtering - coefficients below tol should be discarded
    # This prevents numerical noise from cluttering the operator
    op3 = MatrixOperator(data={("X",): 1e-15, ("P",): 1.0}, tol=1e-12)
    assert ("X",) not in op3.data  # Below tolerance, should be filtered
    assert ("P",) in op3.data      # Above tolerance, should be kept


def test_matrix_operator_arithmetic():
    """
    Test basic arithmetic operations for MatrixOperator.

    Covers the fundamental algebraic operations:
    - Addition: combines operators with coefficient addition
    - Subtraction: combines operators with coefficient subtraction
    - Scalar multiplication: scales all coefficients
    - Negation: reverses sign of all coefficients

    These operations preserve the vector space structure of operators.
    """
    # Create test operators with overlapping and non-overlapping terms
    op1 = MatrixOperator(data={("X",): 1.0, ("P",): 2.0})
    op2 = MatrixOperator(data={("X",): 3.0, ("Y",): 4.0})

    # Addition: coefficients of same terms add, new terms are included
    result = op1 + op2
    assert result.data[("X",)] == 4.0  # 1.0 + 3.0
    assert result.data[("P",)] == 2.0  # from op1 only
    assert result.data[("Y",)] == 4.0  # from op2 only

    # Subtraction: coefficients subtract, maintaining operator structure
    result = op2 - op1
    assert result.data[("X",)] == 2.0   # 3.0 - 1.0
    assert result.data[("P",)] == -2.0  # 0 - 2.0
    assert result.data[("Y",)] == 4.0   # 4.0 - 0

    # Scalar multiplication: all coefficients scaled uniformly
    result = 3 * op1
    assert result.data[("X",)] == 3.0  # 3 * 1.0
    assert result.data[("P",)] == 6.0  # 3 * 2.0

    # Negation: equivalent to multiplication by -1
    result = -op1
    assert result.data[("X",)] == -1.0
    assert result.data[("P",)] == -2.0


def test_matrix_operator_multiplication():
    """
    Test operator multiplication (concatenation).

    Matrix operator multiplication corresponds to concatenating the
    operator sequences. For example: X * P = XP (as a sequence).
    This is fundamental to building higher-degree operators from
    elementary matrix operators.
    """
    # Create operators with multiple terms to test distribution
    op1 = MatrixOperator(data={("X",): 1.0, ("P",): 2.0})
    op2 = MatrixOperator(data={("Y",): 3.0})

    # Multiplication should distribute: (X + 2P) * Y = XY + 2PY
    result = op1 * op2
    assert result.data[("X", "Y")] == 3.0  # 1.0 * 3.0
    assert result.data[("P", "Y")] == 6.0  # 2.0 * 3.0


def test_matrix_operator_power():
    """
    Test operator exponentiation for positive integer powers.

    Powers correspond to repeated multiplication:
    X^2 = X * X = XX (as operator sequence)
    X^3 = X * X * X = XXX

    This is essential for constructing polynomial interactions
    in matrix model Hamiltonians.
    """
    op = MatrixOperator(data={("X",): 1.0})

    # Test positive integer powers
    result = op ** 2
    assert result.data[("X", "X")] == 1.0

    result = op ** 3
    assert result.data[("X", "X", "X")] == 1.0

    # Test error cases - only positive integer powers are meaningful
    with pytest.raises(ValueError):
        op ** -1    # Negative powers not defined for matrix operators

    with pytest.raises(ValueError):
        op ** 1.5   # Fractional powers not defined


def test_matrix_operator_equality():
    """
    Test equality comparison with numerical tolerance.

    Equality is determined up to a tolerance to handle floating-point
    precision issues. This is crucial for numerical stability in
    bootstrap calculations where small rounding errors are common.
    """
    op1 = MatrixOperator(data={("X",): 1.0})
    op2 = MatrixOperator(data={("X",): 1.0 + 1e-15})  # Machine precision difference
    op3 = MatrixOperator(data={("X",): 1.1})          # Significant difference

    assert op1 == op2  # Should be equal within numerical tolerance
    assert op1 != op3  # Should be unequal due to significant difference


def test_matrix_operator_error_handling():
    """
    Test robust error handling for invalid inputs.

    Ensures that the operator classes fail gracefully with informative
    error messages when given invalid data, preventing silent bugs
    in physics calculations.
    """
    # Test invalid coefficient type - must be numeric
    with pytest.raises(TypeError):
        MatrixOperator(data={("X",): "invalid"})

    # Test invalid key type - must be tuple of strings or single string
    with pytest.raises(TypeError):
        MatrixOperator(data={123: 1.0})

    # Test asymmetric addition behavior due to inheritance
    matrix_op = MatrixOperator(data={("X",): 1.0})
    trace_op = SingleTraceOperator(data={("X",): 1.0})

    # This works: base class + derived class → base class result
    result = matrix_op + trace_op
    assert isinstance(result, MatrixOperator)
    assert result.data[("X",)] == 2.0

    # This fails: derived class enforces strict type checking
    with pytest.raises(TypeError, match="Cannot add MatrixOperator to SingleTraceOperator"):
        trace_op + matrix_op


def test_operator_addition_inheritance():
    """
    Test the inheritance-based addition behavior.

    Documents the asymmetric but logical behavior:
    - MatrixOperator + SingleTraceOperator → allowed (inheritance)
    - SingleTraceOperator + MatrixOperator → not allowed (type safety)

    This preserves both usability and type safety.
    """
    matrix_op = MatrixOperator(data={("X",): 1.0, ("P",): 2.0})
    trace_op = SingleTraceOperator(data={("X",): 3.0, ("Y",): 4.0})

    # Base class is permissive with its subclasses
    result = matrix_op + trace_op
    assert isinstance(result, MatrixOperator)  # Result type is base class
    assert result.data[("X",)] == 4.0  # 1.0 + 3.0
    assert result.data[("P",)] == 2.0  # from matrix_op
    assert result.data[("Y",)] == 4.0  # from trace_op

    # Derived class enforces strict typing for safety
    with pytest.raises(TypeError):
        trace_op + matrix_op

    # Same-type addition works fine for both
    matrix_op2 = MatrixOperator(data={("Z",): 5.0})
    trace_op2 = SingleTraceOperator(data={("Z",): 6.0})

    assert isinstance(matrix_op + matrix_op2, MatrixOperator)
    assert isinstance(trace_op + trace_op2, SingleTraceOperator)


def test_single_trace_commutator_onematrix():
    """
    Test single-trace commutator for one-matrix quantum mechanics.

    Validates the Hamiltonian constraint ⟨[H,O]⟩ = 0 for the specific case:
    - O = tr(XP): observable operator
    - H = tr(P²) + tr(X²) + 7·tr(X⁴): single-matrix Hamiltonian

    This test verifies Eq. (8) from the PRL paper on matrix bootstrap
    (PhysRevLett.125.041601), ensuring our commutator calculation
    matches analytical results.

    The commutation relation [P,X] = -i is used.
    """
    # Set up canonical quantum mechanics commutation relations
    matrix_system = MatrixSystem(
        operator_basis=["X", "P"],
        commutation_rules_concise={
            ("P", "X"): -1j,  # [P,X] = -i (canonical momentum-position)
        },
        hermitian_dict={"P": True, "X": True},  # Both operators Hermitian
    )

    # Define the observable and Hamiltonian operators
    OP1 = SingleTraceOperator(data={("X", "P"): 1})  # O = tr(XP)
    OP2 = SingleTraceOperator(
        data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): 7}
    )  # H = tr(P²) + tr(X²) + 7·tr(X⁴)

    # Expected commutator result from analytical calculation
    expected_commutator = SingleTraceOperator(
        data={("P", "P"): 2j, ("X", "X"): -2j, ("X", "X", "X", "X"): -4 * 7j}
    )

    # Test both commutator implementations give same result
    assert matrix_system.single_trace_commutator(OP1, OP2) == expected_commutator
    assert matrix_system.single_trace_commutator2(OP1, OP2) == expected_commutator


def test_single_trace_commutator_twomatrix():
    """
    Test single-trace commutator for two-matrix model.

    Validates commutator calculation for the two-matrix system studied
    in Han et al. This is a more complex example with:
    - Two coordinate matrices X1, X2 and momenta P1, P2
    - Kinetic terms: tr(P1²) + tr(X1²) + tr(P2²) + tr(X2²)
    - Interaction terms with couplings g, h

    Reference: https://github.com/hanxzh94/matrix-bootstrap

    This test ensures our implementation matches established results
    for realistic two-matrix quantum mechanics.
    """
    g, h = 1, 3  # Coupling constants for interaction terms

    # Set up two independent canonical pairs
    matrix_system = MatrixSystem(
        operator_basis=["P1", "X1", "P2", "X2"],
        commutation_rules_concise={
            ("P1", "X1"): -1j,  # [P1, X1] = -i
            ("P2", "X2"): -1j,  # [P2, X2] = -i
            # No cross-commutators: [X1, X2] = [P1, P2] = etc. = 0
        },
        hermitian_dict={"P1": True, "X1": True, "P2": True, "X2": True},
    )

    # Construct the two-matrix Hamiltonian
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
        }
    )

    # Test operator: S2 = tr(X1 P2) - tr(X2 P1)
    S2 = SingleTraceOperator(data={("X1", "P2"): 1, ("X2", "P1"): -1})

    # Expected commutator from analytical calculation
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


def test_zero_single_trace_operator():
    """
    Test behavior with zero single-trace operators.

    Ensures that zero operators (empty or with zero coefficients)
    behave correctly in all algebraic operations. This is important
    for numerical stability and edge case handling.
    """
    # Create zero operator using identity with zero coefficient
    zero = SingleTraceOperator(data={(): 0})
    assert len(zero) == 0  # Zero coefficients should be filtered out

    # Zero coefficient operators should equal the empty operator
    zero_via_coefficient = SingleTraceOperator(data={("P", "P"): 0})
    assert zero_via_coefficient == zero


def test_zero_double_trace_operator():
    """
    Test zero behavior for double-trace operators.

    Verifies that multiplication of zero single-trace operators
    correctly produces zero double-trace operators.
    """
    zero = SingleTraceOperator(data={(): 0})
    # Zero times zero should remain zero, even when promoted to double-trace
    assert zero * zero == zero


def test_single_trace_component_of_double_trace():
    """
    Test extraction of single-trace components from double-trace operators.

    When a double-trace operator has the form tr(1)·tr(O) or tr(O)·tr(1),
    it should be convertible back to the single-trace operator tr(O).
    This is essential for handling the identity operator in calculations.
    """
    # Create a test single-trace operator
    op = SingleTraceOperator(
        data={("P", "P"): 1, ("X", "X"): 1, ("X", "X", "X", "X"): 7}
    )

    # Identity operator
    one = SingleTraceOperator(data={(): 1})

    # Both tr(1)·tr(O) and tr(O)·tr(1) should extract to tr(O)
    assert (one * op).get_single_trace_component() == op
    assert (op * one).get_single_trace_component() == op


def test_get_real_and_imag_parts():
    """
    Test decomposition of operators into real and imaginary parts.

    Complex coefficients in operators can arise from commutators
    (which often produce factors of i). This test ensures we can
    properly separate real and imaginary components for analysis.
    """
    # Test purely real operator
    st_op = SingleTraceOperator(data={"A": 1, "B": 2, "C": 3})
    assert st_op.is_real()

    # Test mixed real/imaginary operator
    st_op = SingleTraceOperator(data={"A": 1, "B": 2, "C": 3j})
    assert not st_op.is_real()

    # Test purely imaginary operator
    st_op = SingleTraceOperator(data={"A": 1j, "B": 2j, "C": 3j})
    assert st_op.is_imag()

    # Test extraction of real and imaginary parts
    st_op = SingleTraceOperator(data={"A": 1j, "B": 2j, "C": 3})
    assert not st_op.is_imag()

    # Verify correct decomposition: op = real_part + i * imag_part
    assert 1j * st_op.get_imag_part() == SingleTraceOperator(data={"A": 1j, "B": 2j})
    assert st_op.get_real_part() == SingleTraceOperator(data={"C": 3})


def test_single_trace_operator_creation():
    """
    Test creation and basic properties of SingleTraceOperator.

    Single-trace operators represent tr(...) expressions which are
    the fundamental observables in matrix models. This test ensures
    basic functionality works correctly.
    """
    # Test creation from dictionary data
    op = SingleTraceOperator(data={("X", "P"): 1.0})
    assert len(op) == 1
    assert op.data[("X", "P")] == 1.0

    # Test string representation contains trace notation
    str_repr = str(op)
    assert "tr(" in str_repr or "<" in str_repr  # Different notation styles


def test_single_trace_operator_arithmetic():
    """
    Test arithmetic operations specific to SingleTraceOperator.

    Single-trace operators form a vector space under addition and
    scalar multiplication, but multiplication promotes to double-trace.
    """
    op1 = SingleTraceOperator(data={("X",): 1.0, ("P",): 2.0})
    op2 = SingleTraceOperator(data={("X",): 3.0, ("Y",): 4.0})

    # Addition works same as for matrix operators
    result = op1 + op2
    assert result.data[("X",)] == 4.0
    assert result.data[("P",)] == 2.0
    assert result.data[("Y",)] == 4.0

    # Multiplication creates double-trace: tr(...)·tr(...)
    result = op1 * op2
    assert isinstance(result, DoubleTraceOperator)


def test_single_trace_operator_conjugation():
    """
    Test Hermitian conjugation of single-trace operators.

    For tr(ABC...), the Hermitian conjugate is tr(...†C†B†A†).
    This respects both the reversal of order and individual conjugations
    according to the hermitian_dict.
    """
    # Set up system with mixed Hermitian/anti-Hermitian operators
    matrix_system = MatrixSystem(
        operator_basis=["X", "P"],
        commutation_rules_concise={("P", "X"): -1j},
        hermitian_dict={"X": True, "P": False}  # X† = X, P† = -P
    )

    # Test Hermitian conjugate of tr(XP)
    op = SingleTraceOperator(data={("X", "P"): 1.0})
    hc_op = matrix_system.hermitian_conjugate(op)

    # (XP)† = P†X† = (-P)X = -PX
    assert hc_op.data[("P", "X")] == -1.0


def test_double_trace_operator_creation():
    """
    Test creation and properties of DoubleTraceOperator.

    Double-trace operators represent products tr(...)·tr(...) which
    appear in operator product expansions and correlation functions.
    """
    # Test creation with various operator combinations
    data = {(("X",), ("P",)): 1.0, (("A", "B"), ("C",)): 2.0}
    op = DoubleTraceOperator(data=data)
    assert len(op) == 2
    assert op.data[(("X",), ("P",))] == 1.0

    # Test string representation shows double-trace structure
    str_repr = str(op)
    assert "tr(" in str_repr  # Should contain trace notation


def test_double_trace_single_trace_component():
    """
    Test extraction of single-trace components from double-trace operators.

    When one trace in tr(A)·tr(B) is the identity tr(1), the result
    should be equivalent to the single-trace tr(A) or tr(B).
    This handles the embedding of single-trace in double-trace space.
    """
    # Test identity in first position: tr(1)·tr(XP) ≡ tr(XP)
    op1 = DoubleTraceOperator(data={((), ("X", "P")): 2.0})
    st_component = op1.get_single_trace_component()
    assert isinstance(st_component, SingleTraceOperator)
    assert st_component.data[("X", "P")] == 2.0

    # Test identity in second position: tr(AB)·tr(1) ≡ tr(AB)
    op2 = DoubleTraceOperator(data={(("A", "B"), ()): 3.0})
    st_component = op2.get_single_trace_component()
    assert st_component.data[("A", "B")] == 3.0


def test_matrix_system_creation():
    """
    Test creation and validation of MatrixSystem.

    MatrixSystem encodes the physical structure: which operators exist,
    their commutation relations, and Hermiticity properties. Proper
    validation prevents inconsistent physics setups.
    """
    # Test valid system creation
    system = MatrixSystem(
        operator_basis=["X", "P"],
        commutation_rules_concise={("P", "X"): -1j},
        hermitian_dict={"X": True, "P": False}
    )
    assert "X" in system.operator_basis
    assert system.hermitian_dict["X"] == True

    # Test validation: hermitian_dict must cover all operators
    with pytest.raises(ValueError):
        MatrixSystem(
            operator_basis=["X", "P"],
            commutation_rules_concise={},
            hermitian_dict={"X": True}  # Missing "P" - should raise error
        )


def test_matrix_system_commutation_expansion():
    """
    Test automatic expansion of commutation relations.

    When [A,B] = C is specified, the system should automatically
    add [B,A] = -C to maintain antisymmetry of commutators.
    This reduces user input requirements and prevents inconsistencies.
    """
    system = MatrixSystem(
        operator_basis=["X", "P"],
        commutation_rules_concise={("P", "X"): -1j},
        hermitian_dict={"X": True, "P": False}
    )

    # Check that antisymmetric rules are automatically added
    assert system.commutation_rules[("P", "X")] == -1j  # As specified
    assert system.commutation_rules[("X", "P")] == 1j   # Auto-generated


def test_complex_coefficients():
    """
    Test handling of complex coefficients throughout the algebra.

    Complex coefficients naturally arise from commutators and
    Hermitian conjugation. All operator classes must handle
    complex arithmetic correctly.
    """
    # Test MatrixOperator with complex coefficients
    op = MatrixOperator(data={("X",): 1 + 2j, ("P",): 3 - 4j})
    assert op.data[("X",)] == 1 + 2j
    assert op.data[("P",)] == 3 - 4j

    # Test SingleTraceOperator complex analysis methods
    st_op = SingleTraceOperator(data={("X",): 2j})
    assert st_op.is_imag()  # Purely imaginary
    assert not st_op.is_real()


def test_tolerance_behavior():
    """
    Test numerical tolerance handling throughout calculations.

    Floating-point arithmetic introduces small errors. Proper tolerance
    handling ensures numerical stability while filtering genuine noise.
    """
    # Test coefficient filtering during creation
    op1 = MatrixOperator(data={("X",): 1e-15, ("P",): 1.0}, tol=1e-12)
    assert ("X",) not in op1.data  # Below tolerance, filtered
    assert ("P",) in op1.data      # Above tolerance, kept

    # Test tolerance in equality comparison
    op2 = MatrixOperator(data={("X",): 1.0})
    op3 = MatrixOperator(data={("X",): 1.0 + 1e-15})  # Machine precision difference
    assert op2 == op3  # Should be equal within tolerance


def test_empty_operators():
    """
    Test behavior with empty (zero) operators.

    Empty operators represent the zero element in the operator algebra.
    They must behave correctly as additive identity and multiplicative
    annihilator to maintain algebraic consistency.
    """
    empty = MatrixOperator(data={})
    non_empty = MatrixOperator(data={("X",): 1.0})

    # Test additive identity: 0 + A = A
    result = empty + non_empty
    assert result.data[("X",)] == 1.0

    # Test multiplicative annihilator: 0 * A = 0
    result = empty * non_empty
    assert len(result) == 0

    # Test zero detection
    assert empty.is_zero()


def test_chained_operations():
    """
    Test complex sequences of algebraic operations.

    Real calculations involve chains of operations. This test ensures
    that operator precedence, associativity, and distributivity
    work correctly in practice.
    """
    op1 = MatrixOperator(data={("X",): 1.0})
    op2 = MatrixOperator(data={("P",): 2.0})
    op3 = MatrixOperator(data={("Y",): 3.0})

    # Test distributivity: (A + B) * C = A*C + B*C
    result = (op1 + op2) * op3
    assert result.data[("X", "Y")] == 3.0  # 1.0 * 3.0
    assert result.data[("P", "Y")] == 6.0  # 2.0 * 3.0

    # Test scalar multiplication: A + 2*B
    result = op1 + 2 * op2
    assert result.data[("X",)] == 1.0
    assert result.data[("P",)] == 4.0


def test_realistic_physics_calculation():
    """
    Test a realistic calculation resembling actual matrix model physics.

    This test constructs a simple two-matrix quantum mechanics system
    similar to those studied in the matrix model bootstrap literature.
    It verifies that our algebra handles realistic Hamiltonians correctly.
    """
    # Create a two-matrix system (like a 2D harmonic oscillator)
    system = MatrixSystem(
        operator_basis=["X1", "X2", "P1", "P2"],
        commutation_rules_concise={
            ("P1", "X1"): -1j,  # [P1, X1] = -i
            ("P2", "X2"): -1j   # [P2, X2] = -i
            # Matrices from different pairs commute: [X1, X2] = 0, etc.
        },
        hermitian_dict={"X1": True, "X2": True, "P1": True, "P2": True}
    )

    # Construct kinetic energy: (1/2) * Σ(Pi² + Xi²)
    kinetic = SingleTraceOperator(data={
        ("P1", "P1"): 0.5,
        ("P2", "P2"): 0.5,
        ("X1", "X1"): 0.5,
        ("X2", "X2"): 0.5
    })

    # Add interaction term: tr(X1 X2 X1 X2)
    interaction = SingleTraceOperator(data={("X1", "X2", "X1", "X2"): 1.0})

    # Build total Hamiltonian: H = H_kinetic + λ * H_interaction
    hamiltonian = kinetic + 0.1 * interaction

    # Verify structure
    assert len(hamiltonian) == 5  # 4 kinetic + 1 interaction term
    assert hamiltonian.data[("P1", "P1")] == 0.5
    assert hamiltonian.data[("X1", "X2", "X1", "X2")] == 0.1

    # Test commutator with simple operator (should be non-zero due to dynamics)
    simple_op = SingleTraceOperator(data={("X1", "P1"): 1.0})
    commutator = system.single_trace_commutator(hamiltonian, simple_op)

    # Commutator should be non-zero (Heisenberg equations of motion)
    assert not commutator.is_zero()


def test_operator_degrees():
    """
    Test degree calculation for operators.

    The degree of an operator term is the number of matrices in the product.
    This is important for organizing operators by complexity and implementing
    truncation schemes in bootstrap calculations.
    """
    # Create operator with terms of various degrees
    op = MatrixOperator(data={
        ("X",): 1.0,           # degree 1: single matrix
        ("X", "P"): 2.0,       # degree 2: two matrices
        ("X", "P", "Y"): 3.0   # degree 3: three matrices
    })

    # Verify degree tracking
    assert 1 in op.degrees
    assert 2 in op.degrees
    assert 3 in op.degrees
    assert op.max_degree == 3


@pytest.mark.parametrize("coeff", [1.0, -2.5, 3.14, 1e-6])
def test_scalar_multiplication_parametrized(coeff):
    """
    Parametrized test for scalar multiplication with various coefficients.

    Ensures scalar multiplication works correctly across a range of
    coefficient values including positive, negative, and small numbers.
    This is essential for building linear combinations of operators.
    """
    op = MatrixOperator(data={("X",): 1.0, ("P",): 2.0})
    result = coeff * op

    # Verify all coefficients are scaled uniformly
    assert result.data[("X",)] == coeff
    assert result.data[("P",)] == 2 * coeff


@pytest.mark.parametrize("power", [1, 2, 3, 4, 5])
def test_operator_powers_parametrized(power):
    """
    Parametrized test for operator powers.

    Tests that X^n produces the correct operator sequence X...X (n times).
    This is crucial for constructing polynomial interactions like tr(X^4)
    that appear in matrix model Hamiltonians.
    """
    op = MatrixOperator(data={("X",): 1.0})
    result = op ** power

    # Should produce single term with coefficient 1.0
    expected_key = tuple(["X"] * power)
    assert result.data[expected_key] == 1.0
    assert len(result) == 1  # Only one term in result
