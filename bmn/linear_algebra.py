from numbers import Number
from typing import Union, Tuple, Dict, Any
import warnings

import numpy as np
from scipy.linalg import null_space, svd
from scipy.sparse import coo_matrix, csc_matrix, spmatrix
from scipy.sparse.linalg import svds
from sparseqr import qr

TOL = 1e-9

SparseMatrix = Union[coo_matrix, csc_matrix, spmatrix]
DenseMatrix = np.ndarray
Matrix = Union[DenseMatrix, SparseMatrix]


def create_sparse_matrix_from_dict(
    index_value_dict: Dict[Tuple[int, int], Number],
    matrix_shape: Tuple[int, int],
    format: str = 'coo'
) -> Union[coo_matrix, csc_matrix]:
    """
    Create a sparse matrix from an index-value dictionary.

    Parameters
    ----------
    index_value_dict : Dict[Tuple[int, int], Number]
        Dictionary mapping (row, col) indices to values:
        {(0, 1): 4, (1, 2): 7, (2, 0): 5, (3, 3): 1}
    matrix_shape : Tuple[int, int]
        Matrix dimensions (rows, cols)
    format : str, optional
        Sparse matrix format ('coo', 'csr', 'csc'), by default 'coo'

    Returns
    -------
    Union[coo_matrix, csc_matrix]
        The sparse matrix

    Raises
    ------
    ValueError
        If indices are out of bounds or format is invalid
    """
    if not index_value_dict:
        # Handle empty dictionary case
        return coo_matrix(matrix_shape) if format == 'coo' else csc_matrix(matrix_shape)

    # Validate matrix shape
    if len(matrix_shape) != 2 or any(dim <= 0 for dim in matrix_shape):
        raise ValueError(f"Invalid matrix shape: {matrix_shape}")

    # Extract indices and values more efficiently
    rows, cols, data = zip(*[
        (row, col, value)
        for (row, col), value in index_value_dict.items()
    ])

    # Validate indices are within bounds
    max_row, max_col = max(rows), max(cols)
    if max_row >= matrix_shape[0] or max_col >= matrix_shape[1]:
        raise ValueError(
            f"Index out of bounds: max indices ({max_row}, {max_col}) "
            f"exceed matrix shape {matrix_shape}"
        )

    # Create sparse matrix
    sparse_matrix = coo_matrix((data, (rows, cols)), shape=matrix_shape)

    # Convert to requested format
    if format == 'csc':
        return sparse_matrix.tocsc()
    elif format == 'csr':
        return sparse_matrix.tocsr()
    elif format == 'coo':
        return sparse_matrix
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_null_space_dense(
    matrix: DenseMatrix,
    tol: float = TOL,
    check_condition: bool = True
) -> np.ndarray:
    """
    Compute the null space of a dense matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of shape (m, n)
    tol : float, optional
        Numerical tolerance, by default TOL
    check_condition : bool, optional
        Whether to verify null space condition, by default True

    Returns
    -------
    np.ndarray
        Null space basis matrix of shape (n, dim(null_space))
        Each column is a null vector: matrix @ null_space = 0

    Raises
    ------
    ValueError
        If null space verification fails
    """
    if matrix.size == 0:
        return np.empty((matrix.shape[1], 0))

    # Compute null space
    null_space_matrix = null_space(matrix)

    if check_condition and null_space_matrix.size > 0:
        # Verify null space condition: M @ null_space ≈ 0
        verification_result = matrix @ null_space_matrix
        violation = np.max(np.abs(verification_result))

        if violation > tol:
            warnings.warn(
                f"Null space condition violated with error {violation:.2e} > {tol:.2e}",
                UserWarning
            )
            if violation > 100 * tol:  # Only raise for severe violations
                raise ValueError(
                    f"Null space condition severely violated: {violation:.2e} > {tol:.2e}"
                )

    return null_space_matrix


def get_row_space_dense(
    matrix: DenseMatrix,
    tol: float = TOL,
    full_matrices: bool = False
) -> np.ndarray:
    """
    Compute the row space basis of a dense matrix using SVD.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of shape (m, n)
    tol : float, optional
        Tolerance for rank determination, by default TOL
    full_matrices : bool, optional
        Whether to compute full SVD matrices, by default False

    Returns
    -------
    np.ndarray
        Row space basis matrix of shape (rank, n)
        Each row spans the row space of the input matrix

    Notes
    -----
    The row space is spanned by the first 'rank' rows of V^T from SVD.
    """
    if matrix.size == 0:
        return np.empty((0, matrix.shape[1]))

    # Perform SVD
    try:
        U, S, Vh = svd(matrix, full_matrices=full_matrices)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"SVD failed: {e}")

    # Determine numerical rank
    rank = np.sum(S > tol)

    if rank == 0:
        return np.empty((0, matrix.shape[1]))

    # Extract row space basis from V^T
    row_space_basis = Vh[:rank, :]

    return row_space_basis


def get_null_space_sparse(
    matrix: SparseMatrix,
    tol: float = TOL,
    check_condition: bool = True
) -> csc_matrix:
    """
    Compute the null space of a sparse matrix using QR decomposition.

    Parameters
    ----------
    matrix : SparseMatrix
        Input sparse matrix of shape (m, n)
    tol : float, optional
        Numerical tolerance, by default TOL
    check_condition : bool, optional
        Whether to verify null space condition, by default True

    Returns
    -------
    csc_matrix
        Null space basis matrix of shape (n, dim(null_space))

    Raises
    ------
    ValueError
        If null space verification fails
    """
    if matrix.nnz == 0:
        return csc_matrix((matrix.shape[1], matrix.shape[1]))

    # QR decomposition of transpose
    try:
        q, _, _, rank = qr(matrix.transpose())
        null_space_matrix = csc_matrix(q)[:, rank:]
    except Exception as e:
        raise RuntimeError(f"QR decomposition failed: {e}")

    if check_condition and null_space_matrix.nnz > 0:
        # Verify null space condition
        verification_result = matrix @ null_space_matrix
        violation = np.max(np.abs(verification_result.data)) if verification_result.nnz > 0 else 0

        if violation > tol:
            warnings.warn(
                f"Sparse null space condition violated: {violation:.2e} > {tol:.2e}",
                UserWarning
            )
            if violation > 100 * tol:
                raise ValueError(
                    f"Sparse null space condition severely violated: {violation:.2e}"
                )

    return null_space_matrix


def get_row_space_sparse(
    matrix: SparseMatrix,
    tol: float = TOL
) -> csc_matrix:
    """
    Compute the row space basis of a sparse matrix using QR decomposition.

    Parameters
    ----------
    matrix : SparseMatrix
        Input sparse matrix of shape (m, n)
    tol : float, optional
        Numerical tolerance, by default TOL

    Returns
    -------
    csc_matrix
        Row space basis matrix of shape (rank, n)

    Raises
    ------
    RuntimeError
        If QR decomposition fails
    AssertionError
        If dimension consistency check fails
    """
    if matrix.nnz == 0:
        return csc_matrix((0, matrix.shape[1]))

    try:
        q, _, _, rank = qr(matrix.transpose())
        row_space_matrix = csc_matrix(q)[:, :rank].transpose()
    except Exception as e:
        raise RuntimeError(f"Sparse QR decomposition failed: {e}")

    # Dimension consistency check
    if row_space_matrix.shape[1] != matrix.shape[1]:
        raise AssertionError(
            f"Dimension mismatch: row space has {row_space_matrix.shape[1]} "
            f"columns but matrix has {matrix.shape[1]}"
        )

    return row_space_matrix


def is_in_row_space_dense(
    matrix: DenseMatrix,
    vector: np.ndarray,
    tol: float = TOL
) -> Tuple[bool, float]:
    """
    Test if a vector lies in the row space of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of shape (m, n)
    vector : np.ndarray
        Test vector of length n
    tol : float, optional
        Numerical tolerance, by default TOL

    Returns
    -------
    Tuple[bool, float]
        (is_in_space, reconstruction_error)

    Raises
    ------
    ValueError
        If dimensions are incompatible
    """
    if matrix.shape[1] != len(vector):
        raise ValueError(
            f"Dimension mismatch: matrix has {matrix.shape[1]} columns "
            f"but vector has {len(vector)} elements"
        )

    if matrix.size == 0:
        return np.allclose(vector, 0, atol=tol), np.linalg.norm(vector)

    # Get row space basis
    row_space_basis = get_row_space_dense(matrix, tol)

    if row_space_basis.size == 0:
        # Empty row space - only zero vector is in it
        error = np.linalg.norm(vector)
        return error <= tol, error

    # Project vector onto row space
    try:
        # Solve least squares: row_space_basis.T @ c = vector
        coefficients, residuals, rank, singular_values = np.linalg.lstsq(
            row_space_basis.T, vector, rcond=None
        )

        # Reconstruct vector from row space
        reconstructed = row_space_basis.T @ coefficients
        reconstruction_error = np.linalg.norm(vector - reconstructed)

        return reconstruction_error <= tol, reconstruction_error

    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Least squares solution failed: {e}")


def is_in_row_space_sparse(
    matrix: SparseMatrix,
    vector: np.ndarray,
    tol: float = TOL
) -> Tuple[bool, float]:
    """Test if a vector lies in the row space of a sparse matrix."""
    if matrix.shape[1] != len(vector):
        raise ValueError(
            f"Dimension mismatch: matrix has {matrix.shape[1]} columns "
            f"but vector has {len(vector)} elements"
        )

    if matrix.nnz == 0:
        error = np.linalg.norm(vector)
        return error <= tol, error

    row_space_basis = get_row_space_sparse(matrix, tol)

    if row_space_basis.nnz == 0:
        error = np.linalg.norm(vector)
        return error <= tol, error

    # Convert to dense for least squares (if not too large)
    if row_space_basis.shape[0] * row_space_basis.shape[1] < 10000:
        row_space_dense = row_space_basis.toarray()
        return is_in_row_space_dense(row_space_dense, vector, tol)  # Remove .T here
    else:
        # For very large matrices, use sparse solvers
        from scipy.sparse.linalg import lsqr
        coefficients = lsqr(row_space_basis.T, vector)[0]
        reconstructed = row_space_basis.T @ coefficients
        error = np.linalg.norm(vector - reconstructed)
        return error <= tol, error


def get_real_coefficients_from_dict(
    coefficients_dict: Dict[Any, Number],
    tol: float = TOL
) -> Dict[Any, float]:
    """
    Extract real coefficients from a dictionary, handling complex values.

    Parameters
    ----------
    coefficients_dict : Dict[Any, Number]
        Dictionary with numeric coefficients
    tol : float, optional
        Tolerance for checking if coefficients are real/imaginary, by default TOL

    Returns
    -------
    Dict[Any, float]
        Dictionary with real coefficients

    Raises
    ------
    ValueError
        If coefficients are neither purely real nor purely imaginary
    """
    if not coefficients_dict:
        return {}

    coefficients = np.array(list(coefficients_dict.values()))

    real_parts = np.real(coefficients)
    imag_parts = np.imag(coefficients)

    is_all_real = np.allclose(coefficients, real_parts, atol=tol)
    is_all_imag = np.allclose(coefficients, 1j * imag_parts, atol=tol)

    if is_all_real:
        return {key: float(np.real(coeff)) for key, coeff in coefficients_dict.items()}
    elif is_all_imag:
        return {key: float(np.imag(coeff)) for key, coeff in coefficients_dict.items()}
    else:
        mixed_indices = []
        for i, coeff in enumerate(coefficients):
            if not (np.isclose(coeff, np.real(coeff), atol=tol) or
                   np.isclose(coeff, 1j * np.imag(coeff), atol=tol)):
                mixed_indices.append(i)

        raise ValueError(
            f"Coefficients are neither all real nor all imaginary. "
            f"Mixed entries at indices: {mixed_indices[:5]}..."
        )


def split_complex_coefficients(
    coefficients_dict: Dict[Any, Number]
) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    """
    Split complex coefficients into real and imaginary parts.

    Parameters
    ----------
    coefficients_dict : Dict[Any, Number]
        Dictionary with potentially complex coefficients

    Returns
    -------
    Tuple[Dict[Any, float], Dict[Any, float]]
        (real_part_dict, imaginary_part_dict)
    """
    real_dict = {}
    imag_dict = {}

    for key, coeff in coefficients_dict.items():
        real_part = float(np.real(coeff))
        imag_part = float(np.imag(coeff))

        if abs(real_part) > TOL:
            real_dict[key] = real_part
        if abs(imag_part) > TOL:
            imag_dict[key] = imag_part

    return real_dict, imag_dict


def matrix_rank(matrix: Matrix, tol: float = TOL) -> int:
    """
    Compute the numerical rank of a matrix.

    Parameters
    ----------
    matrix : Matrix
        Input matrix (dense or sparse)
    tol : float, optional
        Tolerance for rank determination, by default TOL

    Returns
    -------
    int
        Numerical rank of the matrix
    """
    if hasattr(matrix, 'nnz'):  # Sparse matrix
        if matrix.nnz == 0:
            return 0
        # Use SVD for sparse matrices (convert to dense if small enough)
        if matrix.shape[0] * matrix.shape[1] < 10000:
            return matrix_rank(matrix.toarray(), tol)
        else:
            # For large sparse matrices, use iterative methods
            try:
                s = svds(matrix, k=min(matrix.shape)-1, return_singular_vectors=False)
                return int(np.sum(s > tol))
            except:
                return matrix_rank(matrix.toarray(), tol)
    else:  # Dense matrix
        if matrix.size == 0:
            return 0
        s = svd(matrix, compute_uv=False)
        return int(np.sum(s > tol))


def condition_number(matrix: DenseMatrix) -> float:
    """
    Compute the condition number of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix

    Returns
    -------
    float
        Condition number (ratio of largest to smallest singular value)
    """
    if matrix.size == 0:
        return 1.0

    s = svd(matrix, compute_uv=False)
    if len(s) == 0 or s[-1] == 0:
        return np.inf

    return s[0] / s[-1]
