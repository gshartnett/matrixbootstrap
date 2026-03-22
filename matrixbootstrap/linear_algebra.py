from numbers import Number

import numpy as np
from scipy.linalg import (
    null_space,
    svd,
)
from scipy.sparse import (
    coo_matrix,
    csc_matrix,
)
from scipy.sparse.linalg import svds
from sparseqr import qr

import logging

logger = logging.getLogger(__name__)

TOL = 1e-9


def create_sparse_matrix_from_dict(
    index_value_dict: dict[tuple[int, int], Number], matrix_shape: tuple[int, int]
) -> coo_matrix:
    """
    Create a sparse COO-formatted matrix from an index-value dictionary.

    Parameters
    ----------
    index_value_dict : dict[tuple[int, int], Number]
        The index-value dictionary, formatted as in:

        index_value_dict = {
            (0, 1): 4,
            (1, 2): 7,
            (2, 0): 5,
            (3, 3): 1
        }

    matrix_shape: tuple[int, int]
        The matrix shape, required because the matrix is sparse.

    Returns
    -------
    coo_matrix
        The sparse COO matrix.
    """

    # prepare the data
    row_indices = []
    column_indices = []
    data_values = []

    for (row, col), value in index_value_dict.items():
        row_indices.append(row)
        column_indices.append(col)
        data_values.append(value)

    # convert lists to numpy arrays
    # row_indices = np.array(row_indices)
    # column_indices = np.array(column_indices)
    # data_values = np.array(data_values)

    # extract row indices, column indices, and values directly
    rows, cols, data = zip(
        *((row, col, value) for (row, col), value in index_value_dict.items())
    )

    # create the sparse matrix
    sparse_matrix = coo_matrix((data, (rows, cols)), shape=matrix_shape)

    return sparse_matrix


def get_null_space_dense(matrix: np.matrix, tol: float = TOL) -> np.ndarray:
    """
    Get the null space of a rectangular matrix M.

    Parameters
    ----------
    matrix : np.matrix
        The matrix, with shape (m, n)
    tol : float, optional
        A tolerance, by default 1e-10

    Returns
    -------
    np.ndarray
        A matrix of null vectors K, such that
            M . K = 0
        The shape of K is (n, dim(Null(M)))

    Raises
    ------
    ValueError
        _description_
    """
    null_space_matrix = null_space(matrix)
    verification_result = matrix.dot(null_space_matrix)
    violation = np.max(np.abs(verification_result))
    if not violation <= tol:
        raise ValueError(
            f"Warning, null space condition not satisfied, violation = {violation}."
        )
    return null_space_matrix


def get_row_space_dense(matrix: np.ndarray, tol: float = TOL) -> np.ndarray:

    # perform SVD on the matrix
    U, S, Vh = svd(matrix)
    # U, S, Vh = svds(matrix, k=min(matrix.shape[0], matrix.shape[1]), solver='propack')

    # determine rank of matrix
    rank = np.sum(
        S > tol
    )  # Consider singular values greater than a small threshold as non-zero

    # extract basis of the row space from the top rank rows of Vt
    row_space_basis = Vh[:rank, :]

    return row_space_basis


def get_null_space_sparse(matrix, tol: float = TOL):
    """
    Computes the right null space of a sparse matrix.
    Arguments:
        mat (sparse matrix): the matrix to compute, of shape (M, N)
    Returns:
        null (sparse matrix): a basis of the null space, of shape (N, K)
    """
    logger.debug("get_null_space_sparse: input shape %s", matrix.shape)
    q, _, _, rank = qr(matrix.transpose())
    null_space_matrix = csc_matrix(q)[:, rank:]
    logger.debug(
        "get_null_space_sparse: null space shape %s (rank=%d)", null_space_matrix.shape, rank
    )
    verification_result = matrix @ null_space_matrix
    violation = np.max(np.abs(verification_result))
    if not violation <= tol:
        raise ValueError(
            f"Warning, null space condition not satisfied, violation = {violation}."
        )
    return null_space_matrix


def get_row_space_sparse(matrix, tol: float = TOL):
    """
    Computes the row space of a sparse matrix.
    Arguments:
            mat (sparse matrix): the matrix to compute, of shape (M, N)
    Returns:
            r (sparse matrix): a basis of the row space, of shape (K, N)
    """
    logger.debug("get_row_space_sparse: input shape %s", matrix.shape)
    q, _, _, rank = qr(matrix.transpose())
    r = csc_matrix(q)[:, :rank].transpose()
    assert r.shape[1] == matrix.shape[1], "dimension mismatch"
    logger.debug("get_row_space_sparse: row space shape %s (rank=%d)", r.shape, rank)
    return r


def is_in_row_space_dense(matrix: np.ndarray, vector: np.ndarray) -> bool:
    if matrix.shape[1] != len(vector):
        raise ValueError("Error, dimension mismatch.")
    row_space_matrix = get_row_space_dense(matrix)

    # Project the vector v onto the row space
    # Solve the linear system row_space_basis.T * c = v to find the coefficients c
    c, residuals, rank, s = np.linalg.lstsq(row_space_matrix.T, vector, rcond=None)

    # Reconstruct v from the row space basis using the coefficients c
    v_projected = row_space_matrix.T @ c

    return np.allclose(v_projected, vector)


def get_real_coefficients_from_dict(x: dict):
    coefficients = np.asarray(list(x.values()))
    all_real = np.allclose(np.real(coefficients), coefficients)
    all_imag = np.allclose(1j * np.imag(coefficients), coefficients)
    if not (all_real or all_imag):
        raise ValueError("Warning, coefficients are not all real or imaginary.")
    if all_real:
        return {key: np.real(coeff) for key, coeff in x.items()}
    if all_imag:
        return {key: np.imag(coeff) for key, coeff in x.items()}
