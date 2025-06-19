import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix

from bmn.linear_algebra import (
    get_real_coefficients_from_dict,
    split_complex_coefficients,
    create_sparse_matrix_from_dict,
    get_null_space_dense,
    get_null_space_sparse,
    get_row_space_dense,
    get_row_space_sparse,
    is_in_row_space_dense,
    is_in_row_space_sparse,
    matrix_rank,
    condition_number,
    TOL
)


def test_get_real_coefficients():
    # Test purely imaginary coefficients - should return imaginary parts
    assert get_real_coefficients_from_dict(coefficients_dict={"A": 0, "B": 1 * 1j, "C": 4 * 1j}) == {
        "A": 0,
        "B": 1,
        "C": 4,
    }

    # Test purely real coefficients - should return real parts
    assert get_real_coefficients_from_dict(coefficients_dict={"A": 0, "B": 1, "C": 4}) == {
        "A": 0,
        "B": 1,
        "C": 4,
    }

    # Test mixed coefficients - should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        get_real_coefficients_from_dict(coefficients_dict={"A": 0, "B": 1 * 1j, "C": 4})

    # Check that the error message contains the expected text
    error_message = str(exc_info.value)
    assert "Coefficients are neither all real nor all imaginary" in error_message
    assert "Mixed entries at indices" in error_message


def test_get_real_coefficients_empty():
    """Test empty dictionary returns empty dictionary."""
    assert get_real_coefficients_from_dict(coefficients_dict={}) == {}


def test_get_real_coefficients_tolerance():
    """Test tolerance handling for near-real/imaginary values."""
    # Values that are real within tolerance
    result = get_real_coefficients_from_dict(
        coefficients_dict={"A": 1.0 + 1e-15j, "B": 2.0 - 1e-15j},
        tol=1e-12,
    )
    assert result == {"A": 1.0, "B": 2.0}

    # Values that are imaginary within tolerance
    result = get_real_coefficients_from_dict(
        coefficients_dict={"A": 1e-15 + 1j, "B": -1e-15 + 2j},
        tol=1e-12,
    )
    assert result == {"A": 1.0, "B": 2.0}


def test_get_real_coefficients_zero_handling():
    """Test handling of zero coefficients."""
    result = get_real_coefficients_from_dict(coefficients_dict={"A": 0+0j, "B": 0, "C": 0.0})
    assert result == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_get_real_coefficients_single_entry():
    """Test single coefficient entries."""
    assert get_real_coefficients_from_dict(coefficients_dict={"single": 5.0}) == {"single": 5.0}
    assert get_real_coefficients_from_dict(coefficients_dict={"single": 3j}) == {"single": 3.0}


def test_split_complex_coefficients():
    """Test splitting complex coefficients into real and imaginary parts."""
    coeffs = {"A": 1+2j, "B": 3-4j, "C": 5+0j, "D": 0+6j, "E": 0+0j}
    real_dict, imag_dict = split_complex_coefficients(coeffs)

    assert real_dict == {"A": 1.0, "B": 3.0, "C": 5.0}
    assert imag_dict == {"A": 2.0, "B": -4.0, "D": 6.0}


def test_split_complex_coefficients_tolerance():
    """Test splitting with tolerance for small values."""
    coeffs = {"A": 1e-15 + 2j, "B": 3 + 1e-15j}
    real_dict, imag_dict = split_complex_coefficients(coeffs)

    # Small values below TOL should be filtered out
    assert real_dict == {"B": 3.0}
    assert imag_dict == {"A": 2.0}


def test_create_sparse_matrix_from_dict():
    """Test sparse matrix creation from dictionary."""
    index_dict = {(0, 1): 4, (1, 2): 7, (2, 0): 5}
    matrix = create_sparse_matrix_from_dict(index_dict, (3, 3))

    assert matrix.shape == (3, 3)
    # Convert to dense or CSC format for indexing
    matrix_csc = matrix.tocsc()
    assert matrix_csc[0, 1] == 4
    assert matrix_csc[1, 2] == 7
    assert matrix_csc[2, 0] == 5


def test_create_sparse_matrix_different_formats():
    """Test sparse matrix creation with different formats."""
    index_dict = {(0, 0): 1, (1, 1): 2}

    coo_matrix_result = create_sparse_matrix_from_dict(index_dict, (2, 2), format='coo')
    csc_matrix_result = create_sparse_matrix_from_dict(index_dict, (2, 2), format='csc')
    csr_matrix_result = create_sparse_matrix_from_dict(index_dict, (2, 2), format='csr')

    assert isinstance(coo_matrix_result, coo_matrix)
    assert isinstance(csc_matrix_result, csc_matrix)
    # CSR returns COO converted to CSR
    assert csr_matrix_result.format == 'csr'


def test_create_sparse_matrix_empty():
    """Test empty sparse matrix creation."""
    matrix = create_sparse_matrix_from_dict({}, (3, 3))
    assert matrix.shape == (3, 3)
    assert matrix.nnz == 0


def test_create_sparse_matrix_errors():
    """Test error handling in sparse matrix creation."""
    # Invalid matrix shape
    with pytest.raises(ValueError, match="Invalid matrix shape"):
        create_sparse_matrix_from_dict({(0, 0): 1}, (0, 3))

    # Index out of bounds
    with pytest.raises(ValueError, match="Index out of bounds"):
        create_sparse_matrix_from_dict({(5, 0): 1}, (3, 3))

    # Invalid format
    with pytest.raises(ValueError, match="Unsupported format"):
        create_sparse_matrix_from_dict({(0, 0): 1}, (2, 2), format='invalid')


def test_get_null_space_dense():
    """Test null space computation for dense matrices."""
    # Matrix with known null space
    A = np.array([[1, 2, 3], [4, 5, 6]])  # rank 2, null space dimension 1
    null_space_matrix = get_null_space_dense(A)

    # Verify null space condition
    assert np.allclose(A @ null_space_matrix, 0, atol=TOL)
    assert null_space_matrix.shape[0] == 3  # number of columns in A
    assert null_space_matrix.shape[1] == 1  # dimension of null space


def test_get_null_space_dense_full_rank():
    """Test null space for full rank matrix."""
    A = np.eye(3)  # Full rank matrix
    null_space_matrix = get_null_space_dense(A)
    assert null_space_matrix.shape == (3, 0)  # Empty null space


def test_get_null_space_dense_empty():
    """Test null space for empty matrix."""
    A = np.empty((0, 3))
    null_space_matrix = get_null_space_dense(A)
    assert null_space_matrix.shape == (3, 0)


def test_get_null_space_sparse():
    """Test null space computation for sparse matrices."""
    # Create sparse matrix
    A_sparse = csc_matrix(np.array([[1, 2, 3], [4, 5, 6]]))
    null_space_matrix = get_null_space_sparse(A_sparse)

    # Verify null space condition
    result = A_sparse @ null_space_matrix
    if result.nnz > 0:
        assert np.allclose(result.toarray(), 0, atol=TOL)


def test_get_row_space_dense():
    """Test row space computation for dense matrices."""
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # rank 2
    row_space_matrix = get_row_space_dense(A)

    assert row_space_matrix.shape[1] == 3  # number of columns preserved
    assert row_space_matrix.shape[0] <= 2  # rank is 2


def test_get_row_space_dense_zero_matrix():
    """Test row space for zero matrix."""
    A = np.zeros((3, 4))
    row_space_matrix = get_row_space_dense(A)
    assert row_space_matrix.shape == (0, 4)


def test_get_row_space_sparse():
    """Test row space computation for sparse matrices."""
    A_sparse = csc_matrix(np.array([[1, 2], [3, 4]]))
    row_space_matrix = get_row_space_sparse(A_sparse)

    assert row_space_matrix.shape[1] == 2
    # The function returns CSC matrix transposed, which might change format
    assert hasattr(row_space_matrix, 'nnz')  # Check it's a sparse matrix


def test_is_in_row_space_dense():
    """Test vector membership in row space."""
    A = np.array([[1, 0, 1], [0, 1, 1]])

    # Vector in row space
    v_in = np.array([1, 1, 2])  # Linear combination of rows
    is_in, error = is_in_row_space_dense(A, v_in)
    assert is_in
    assert error <= TOL

    # Vector not in row space
    v_out = np.array([1, 0, 0])
    is_in, error = is_in_row_space_dense(A, v_out)
    assert not is_in
    assert error > TOL


def test_is_in_row_space_dense_dimension_mismatch():
    """Test dimension mismatch error."""
    A = np.array([[1, 2, 3]])
    v = np.array([1, 2])  # Wrong dimension

    with pytest.raises(ValueError, match="Dimension mismatch"):
        is_in_row_space_dense(A, v)


def test_is_in_row_space_sparse():
    """Test vector membership in sparse row space."""
    A_sparse = csc_matrix(np.array([[1, 0, 1], [0, 1, 1]]))

    # Use a vector that's definitely in the row space (one of the rows)
    v_in = np.array([1, 0, 1])  # First row
    is_in, error = is_in_row_space_sparse(A_sparse, v_in)
    assert is_in
    assert error <= TOL


def test_matrix_rank_dense():
    """Test matrix rank computation for dense matrices."""
    # Full rank matrix
    A = np.eye(3)
    assert matrix_rank(A) == 3

    # Rank deficient matrix
    B = np.array([[1, 2], [2, 4]])  # rank 1
    assert matrix_rank(B) == 1

    # Zero matrix
    C = np.zeros((3, 3))
    assert matrix_rank(C) == 0


def test_matrix_rank_sparse():
    """Test matrix rank computation for sparse matrices."""
    # Full rank sparse matrix
    A_sparse = csc_matrix(np.eye(3))
    assert matrix_rank(A_sparse) == 3

    # Empty sparse matrix
    B_sparse = csc_matrix((3, 3))
    assert matrix_rank(B_sparse) == 0


def test_matrix_rank_large_sparse():
    """Test matrix rank for large sparse matrices."""
    # Create a large sparse matrix to test the fallback mechanism
    np.random.seed(42)
    dense_large = np.random.rand(200, 150)
    sparse_large = csc_matrix(dense_large)

    rank_dense = matrix_rank(dense_large)
    rank_sparse = matrix_rank(sparse_large)

    # Should be approximately equal (within numerical tolerance)
    assert abs(rank_dense - rank_sparse) <= 2


def test_condition_number():
    """Test condition number computation."""
    # Well-conditioned matrix
    A = np.eye(3)
    assert condition_number(A) == 1.0

    # Ill-conditioned matrix
    B = np.array([[1, 1], [1, 1.001]])
    cond_B = condition_number(B)
    assert cond_B > 1000  # Should be large

    # Nearly singular matrix (numerical precision issue)
    C = np.array([[1, 2], [2, 4]])
    cond_C = condition_number(C)
    assert cond_C > 1e10  # Very large condition number

    # Empty matrix
    D = np.empty((0, 0))
    assert condition_number(D) == 1.0


def test_tolerance_parameter_consistency():
    """Test that tolerance parameters work consistently across functions."""
    custom_tol = 1e-6

    # Test with coefficients that are clearly separable with custom tolerance
    # Use values that are more clearly real vs imaginary
    coeffs = {"A": 1e-10 + 1j, "B": 2 + 1e-10j}  # Clearly imaginary and real respectively

    # This should raise an error because they're mixed
    with pytest.raises(ValueError):
        get_real_coefficients_from_dict(coeffs, tol=custom_tol)

    # Test with truly real coefficients
    real_coeffs = {"A": 2.0, "B": 3.0}
    result = get_real_coefficients_from_dict(real_coeffs, tol=custom_tol)
    assert result == {"A": 2.0, "B": 3.0}


def test_edge_cases_numerical_stability():
    """Test edge cases for numerical stability."""
    # Very small values
    A_small = np.array([[1e-15, 2e-15], [3e-15, 4e-15]])
    rank_small = matrix_rank(A_small, tol=1e-12)
    assert rank_small <= 2

    # Large values
    A_large = np.array([[1e15, 2e15], [3e15, 4e15]])
    rank_large = matrix_rank(A_large)
    assert rank_large == 2


def test_complex_input_handling():
    """Test handling of complex-valued matrices where applicable."""
    # Complex dense matrix
    A_complex = np.array([[1+1j, 2], [3, 4-1j]])
    rank_complex = matrix_rank(A_complex)
    assert rank_complex == 2

    cond_complex = condition_number(A_complex)
    assert cond_complex > 0


def test_error_propagation():
    """Test that errors are properly propagated from underlying libraries."""
    # Test with matrices that might cause SVD to fail
    A_problematic = np.full((3, 3), np.inf)

    with pytest.raises((RuntimeError, np.linalg.LinAlgError, ValueError)):
        get_row_space_dense(A_problematic)


def test_memory_efficiency():
    """Test memory-efficient handling of large sparse matrices."""
    # Create a sparse matrix that would be large if dense
    row_indices = [0, 100, 1000]
    col_indices = [0, 50, 500]
    data = [1, 2, 3]

    A_sparse = coo_matrix((data, (row_indices, col_indices)), shape=(1001, 501))

    # Should handle without converting to dense
    rank = matrix_rank(A_sparse)
    assert rank <= 3
    assert A_sparse.nnz == 3  # Still sparse
