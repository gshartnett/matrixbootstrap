import logging
import time
from typing import Optional

import cvxpy as cp
import numpy as np
import scipy.linalg
import scipy.sparse as sparse
from scipy.linalg import ishermitian
from scipy.sparse import csr_matrix

from matrixbootstrap.algebra import SingleTraceOperator
from matrixbootstrap.bootstrap import BootstrapSystem
from matrixbootstrap.linear_algebra import get_null_space_dense
from matrixbootstrap.solver_trustregion import (
    get_quadratic_constraint_vector_sparse as get_quadratic_constraint_vector,
)

logger = logging.getLogger(__name__)


def sdp_minimize(
    linear_objective_vector: np.ndarray,
    bootstrap_table_sparse: csr_matrix,
    linear_inhomogeneous_eq: tuple[csr_matrix, np.ndarray],
    linear_inhomogeneous_penalty: Optional[tuple[csr_matrix, np.ndarray]] = None,
    radius: float = np.inf,
    maxiters: int = 2500,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    eps_infeas: float = 1e-7,
    reg: float = 1e-4,
    verbose: bool = False,
    cvxpy_solver: str = "SCS",
) -> tuple[bool, str, np.ndarray]:
    """
    Performs the following SDP minimization over the vector variable x:

    A linear objective v^T x is minimized, subject to the constraints
        - The bootstrap matrix M(x) is positive semi-definite
        - The linear inhomogeneous equation A x = b is satisfied
        - A second linear inhomgeneous equation A' x = b' is imposed as a penalty
        in the objective function
        - The variable x is constrained to lie within a radius-R ball of an initial
        variable `init`

    Parameters
    ----------
    linear_objective_vector : np.ndarray
        A vector determining the objective function v^T x
    bootstrap_table_sparse : csr_matrix
        The bootstrap table (represented as a sparse CSR matrix)
    linear_inhomogeneous_eq : tuple[csr_matrix, np.ndarray]
        A tuple (A, b) of a matrix A and a vector b, which together
        define a linear inhomogeneous equation of the form A x = b.
        This equation will be imposed directly as a constraint.
    linear_inhomogeneous_penalty : tuple[csr_matrix, np.ndarra]
        A tuple (A, b) of a matrix A and a vector b, which together
        define a linear inhomogeneous equation of the form A x = b.
        This equation will be imposed indirectly as a penalty term
        added to the objective function.
    init : np.ndarray
        An initial value for the variable vector x.
    radius : float, optional
        The maximum allowable change in the initial vector, by default np.inf
    maxiters : int, optional
        The maximum number of iterations for the SDP optimization, by default 10_000
    eps : float, optional
        A tolerance variable for the SDP optimization, by default 1e-4
    reg : float, optional
        A regularization coefficient controlling the relative weight of the
        penalty term, by default 1e-4
    verbose : bool, optional
        An optional boolean flag used to set the verbosity, by default True

    Returns
    -------
    tuple[bool, str, np.ndarray]
        A tuple containing
            a boolean variable, where True corresponds to a successful execution of the optimization
            a str containing the optimization status
            a numpy array with the final, optimized vector
    """
    # CVXPY method (only SCS is supported for this particular convex problem)
    if cvxpy_solver == "SCS":
        solver = cp.SCS
    elif cvxpy_solver == "ECOS":
        solver = cp.ECOS
    elif cvxpy_solver == "OSQP":
        solver = cp.OSQP
    elif cvxpy_solver == "CLARABEL":
        solver = cp.CLARABEL
    elif cvxpy_solver == "MOSEK":
        solver = cp.MOSEK
    else:
        raise NotImplementedError

    # set-up
    matrix_dim = int(np.sqrt(bootstrap_table_sparse.shape[0]))
    num_variables = bootstrap_table_sparse.shape[1]
    param = cp.Variable(num_variables)  # declare the cvxpy param

    # build the constraints
    # 1. the PSD bootstrap constraint(s)
    # 2. A @ param == 0
    # 3. ||param||_2 <= radius

    # If the n x n bootstrap table is complex, split the corresponding bootstrap matrix
    # into a (2n, 2n) real matrix, rather than a (n, n) complex matrix.
    # Note: based on the param vector, it could be the case that a complex-valued bootstrap
    # table gives rise to a real bootstrap matrix - e.g., if odd-degree expectation values
    # vanish for the given param vector.
    # cvxpy's C++ backend requires contiguous float arrays; complex sparse matrices
    # produce non-contiguous views via .real/.imag. Cast explicitly to float64.
    if np.max(np.abs(bootstrap_table_sparse.imag)) > 1e-10:
        logger.debug("Mapping complex bootstrap matrix to real")
        bootstrap_table_sparse_real = bootstrap_table_sparse.real.astype(np.float64)
        bootstrap_table_sparse_imag = bootstrap_table_sparse.imag.astype(np.float64)
        matrix_real = cp.reshape(
            bootstrap_table_sparse_real @ param, (matrix_dim, matrix_dim), order="F"
        )
        matrix_imag = cp.reshape(
            bootstrap_table_sparse_imag @ param, (matrix_dim, matrix_dim), order="F"
        )
        matrix_block = cp.bmat(
            [[matrix_real, -matrix_imag], [matrix_imag, matrix_real]]
        )
        constraints = [matrix_block >> 0]
    else:
        constraints = [
            cp.reshape(
                bootstrap_table_sparse.real.astype(np.float64) @ param,
                (matrix_dim, matrix_dim),
                order="F",
            )
            >> 0
        ]

    constraints += [linear_inhomogeneous_eq[0] @ param == linear_inhomogeneous_eq[1]]
    constraints += [cp.norm(param) <= radius]

    # the loss to minimize
    if linear_objective_vector is not None:
        loss = linear_objective_vector @ param

        # add possible penalty / regularization terms
        # Ax=b penalty
        if linear_inhomogeneous_penalty is not None:
            _ = cp.norm(
                linear_inhomogeneous_penalty[0] @ param
                - linear_inhomogeneous_penalty[1]
            )

        # l2 norm on param vector
        loss += reg * cp.norm(param)

    else:
        loss = cp.norm(param)

    # solve the optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints)
    if cvxpy_solver == "SCS":
        prob.solve(
            verbose=verbose,
            max_iters=maxiters,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_infeas=eps_infeas,
            solver=solver,
        )
    elif cvxpy_solver == "CLARABEL":
        prob.solve(
            verbose=verbose,
            max_iter=maxiters,
            tol_gap_abs=eps_abs,
            tol_gap_rel=eps_rel,
            tol_feas=eps_abs,
            static_regularization_constant=1e-4,
            solver=solver,
        )
    elif cvxpy_solver == "MOSEK":
        prob.solve(
            verbose=verbose,
            # max_iters=maxiters,
            # eps_abs=eps_abs,
            # eps_rel=eps_rel,
            # eps_infeas=eps_infeas,
            solver=solver,
            accept_unknown=True,  # https://www.cvxpy.org/tutorial/solvers/index.html
            mosek_params={
                # "MSK_DPAR_OPTIMIZER_MAX_TIME": 100,
                # "MSK_DPAR_BASIS_TOL_S": 1e-8,
                # "MSK_DPAR_BASIS_TOL_X": 1e-8,
                # "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
                # "MSK_IPAR_SIM_MAX_ITERATIONS": 30_000_000,
                #'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL',
            },
        )

    if param.value is None:
        raise ValueError("sdp_minimize failed, None value returned.")

    # log information on the extent to which the constraints are satisfied
    ball_constraint = np.linalg.norm(param.value)
    violation_of_linear_constraints = np.linalg.norm(
        linear_inhomogeneous_eq[0] @ param.value - linear_inhomogeneous_eq[1]
    )
    min_bootstrap_eigenvalue = np.linalg.eigvalsh(
        (bootstrap_table_sparse @ param.value).reshape(matrix_dim, matrix_dim)
    )[0]
    logger.debug("sdp_minimize status (maxiters=%d): %s", maxiters, prob.status)
    logger.debug("sdp_minimize ||A x - b||: %.4e", violation_of_linear_constraints)
    logger.debug(
        "sdp_minimize bootstrap matrix min eigenvalue: %.4e", min_bootstrap_eigenvalue
    )

    optimization_result = {
        "solver": cvxpy_solver,
        "prob.status": prob.status,
        "prob.value": prob.value,
        "maxiters_cvxpy": maxiters,
        "||x||": ball_constraint,
        "violation_of_linear_constraints": violation_of_linear_constraints,
        "min_bootstrap_eigenvalue": min_bootstrap_eigenvalue,
    }

    return param.value, optimization_result


# ---------------------------------------------------------------------------
# Direct SCS helpers — bypass CVXPY canonicalization (which takes 6+ min for
# large SDP with sparse data due to Python-level coefficient extraction).
# ---------------------------------------------------------------------------


def _scs_complex_psd_rows(
    bt_real: csr_matrix, bt_imag: csr_matrix, d: int
) -> csr_matrix:
    """Build SCS PSD cone rows for the (2d×2d) real bmat [[M_r,-M_i],[M_i,M_r]].

    Returns a sparse (D*(D+1)//2, n_params) matrix where D=2d.  Each row k
    corresponds to the k-th entry of svec(M_block) in SCS column-major
    lower-triangular order (diagonal entries unscaled, off-diagonals × sqrt(2)).

    When used in the SCS constraint  s = b - A@x  with b=0 and A = -result,
    the cone variable s = result @ x = svec(M_block(x)) is forced into the PSD
    cone, which means the 2d×2d bootstrap block is PSD.
    """
    D = 2 * d
    sqrt2 = np.sqrt(2.0)
    rows: list[csr_matrix] = []
    for j in range(D):
        for i in range(j, D):
            scale = sqrt2 if i != j else 1.0
            if j < d and i < d:  # upper-left: M_real[i,j]
                row = bt_real[j * d + i, :]
            elif j < d:  # lower-left: M_imag[i-d,j]
                row = bt_imag[j * d + (i - d), :]
            else:  # lower-right: M_real[i-d,j-d]
                row = bt_real[(j - d) * d + (i - d), :]
            rows.append(row.multiply(scale) if scale != 1.0 else row)
    return sparse.vstack(rows, format="csr")


def _scs_real_psd_rows(bt_real: csr_matrix, d: int) -> csr_matrix:
    """Build SCS PSD cone rows for a real d×d bootstrap matrix."""
    sqrt2 = np.sqrt(2.0)
    rows: list[csr_matrix] = []
    for j in range(d):
        for i in range(j, d):
            scale = sqrt2 if i != j else 1.0
            row = bt_real[j * d + i, :]
            rows.append(row.multiply(scale) if scale != 1.0 else row)
    return sparse.vstack(rows, format="csr")


# ---------------------------------------------------------------------------
# Module-level cache for ADMM Gram matrix / Cholesky factor.
# All configs sharing the same bootstrap tables (same nu only changes c)
# can reuse the same factor — saves ~30s of setup per config.
# ---------------------------------------------------------------------------
_ADMM_CACHE: dict = {}


def _scs_vec_sym(M: np.ndarray) -> np.ndarray:
    """Vectorize symmetric DxD matrix in SCS column-major lower-tri format.

    SCS PSD cone uses column-major lower-triangular storage: entries are
    listed column by column (j=0,1,...; i=j,...,D-1 within each column).
    Diagonal entries unscaled; off-diagonal entries multiplied by sqrt(2).

    For a symmetric matrix, this equals M[triu_r, triu_c] in triu_indices order.
    (triu_indices visits upper-tri in row-major, which for symmetric M gives
    the same values as column-major lower-tri.)
    """
    D = M.shape[0]
    tri_r, tri_c = np.triu_indices(D, k=0)
    v = M[tri_r, tri_c].copy()
    v[tri_r != tri_c] *= np.sqrt(2.0)
    return v


def _scs_unvec_sym(v: np.ndarray, D: int) -> np.ndarray:
    """Reconstruct symmetric DxD matrix from SCS column-major lower-tri vectorization."""
    M = np.zeros((D, D), dtype=np.float64)
    tri_r, tri_c = np.triu_indices(D, k=0)
    vals = v.copy()
    vals[tri_r != tri_c] /= np.sqrt(2.0)
    M[tri_r, tri_c] = vals
    M[tri_c, tri_r] = M[tri_r, tri_c]
    return M


def _admm_proj_psd_blocks(z: np.ndarray, psd_dims: list) -> np.ndarray:
    """Project z (stacked SCS-vectorized symmetric blocks) onto PSD^K cone.

    psd_dims[k] is the block-matrix dimension D_k (2d for complex, d for real).
    Each block occupies D_k*(D_k+1)//2 entries in z.
    """
    result = np.empty_like(z)
    offset = 0
    for D in psd_dims:
        n_k = D * (D + 1) // 2
        M = _scs_unvec_sym(z[offset : offset + n_k], D)
        eigs, vecs = np.linalg.eigh(M)
        M_proj = (vecs * np.maximum(eigs, 0.0)) @ vecs.T
        result[offset : offset + n_k] = _scs_vec_sym(M_proj)
        offset += n_k
    return result


def _admm_min_psd_eig(z: np.ndarray, psd_dims: list) -> float:
    """Return minimum eigenvalue across all PSD blocks in z."""
    min_eig = np.inf
    offset = 0
    for D in psd_dims:
        n_k = D * (D + 1) // 2
        M = _scs_unvec_sym(z[offset : offset + n_k], D)
        eig = float(np.linalg.eigvalsh(M)[0])
        if eig < min_eig:
            min_eig = eig
        offset += n_k
    return min_eig


def _build_null_psd_rows(
    bt_real: csr_matrix,
    bt_imag: csr_matrix,
    d: int,
    ns: np.ndarray,
    pp: np.ndarray,
    has_imag: bool,
) -> tuple:
    """Build dense A_null rows and b_psd for one PSD block (null-space param).

    Returns (A_null, b_psd) where A_null is (n_rows, n_null) dense and
    b_psd is (n_rows,).  n_rows = D*(D+1)//2 with D=2d (complex) or D=d (real).

    Row ordering: lower-triangular column-major of D×D bmat, with sqrt(2)
    scaling for off-diagonal entries (SCS PSD vectorization).
    """
    sqrt2 = np.sqrt(2.0)
    n_null = ns.shape[1]

    if has_imag:
        D = 2 * d
        # sparse @ dense — O(nnz × n_null) via scipy C loop
        proj_r = bt_real.dot(ns)  # (d^2, n_null) float64
        proj_i = bt_imag.dot(ns)  # (d^2, n_null) float64
        const_r = bt_real.dot(pp)  # (d^2,)
        const_i = bt_imag.dot(pp)  # (d^2,)
    else:
        D = d
        proj_r = bt_real.dot(ns)
        proj_i = None
        const_r = bt_real.dot(pp)
        const_i = None

    n_rows = D * (D + 1) // 2
    A_null_block = np.empty((n_rows, n_null), dtype=np.float64)
    b_psd_block = np.empty(n_rows, dtype=np.float64)

    k = 0
    for j in range(D):
        for i in range(j, D):
            scale = sqrt2 if i != j else 1.0
            if has_imag:
                if j < d and i < d:
                    pr = proj_r[j * d + i, :]
                    cb = const_r[j * d + i]
                elif j < d:
                    pr = proj_i[j * d + (i - d), :]
                    cb = const_i[j * d + (i - d)]
                else:
                    pr = proj_r[(j - d) * d + (i - d), :]
                    cb = const_r[(j - d) * d + (i - d)]
            else:
                pr = proj_r[j * d + i, :]
                cb = const_r[j * d + i]
            if scale != 1.0:
                A_null_block[k, :] = pr * scale
                b_psd_block[k] = cb * scale
            else:
                A_null_block[k, :] = pr
                b_psd_block[k] = cb
            k += 1

    return A_null_block, b_psd_block


def _sdp_admm_blas(
    c_null: np.ndarray,
    A_null: np.ndarray,
    b_psd: np.ndarray,
    psd_dims: list,
    maxiters: int = 50000,
    rho: float = 1.0,
    alpha: float = 1.5,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    log_interval: int = 500,
    cache_key: Optional[tuple] = None,
) -> tuple:
    """ADMM SDP solver using BLAS Cholesky — no QDLDL, no CVXPY.

    Uses over-relaxation (alpha=1.5, same as SCS default) to damp oscillations.
    Without over-relaxation the ADMM cycles and r_primal does not decrease.

    Solves:
        min  c_null^T x_null
        s.t. A_null @ x_null + b_psd ∈ PSD^K

    Uses the standard 2-block ADMM splitting:
        z = A_null @ x_null + b_psd  (target slack)
        s = proj_PSD(z)              (PSD slack)
    with dual variable y (for s - z = 0).

    The x-update solves (rho * G) x = rhs, where G = A_null^T A_null.
    G is computed once via numpy DGEMM and cached across calls with the
    same cache_key.

    Returns (x_null, status_str).
    """
    n_null = A_null.shape[1]
    n_psd = A_null.shape[0]

    # --- Gram matrix + Cholesky (cached across configs) ---
    if cache_key is not None and cache_key in _ADMM_CACHE:
        F = _ADMM_CACHE[cache_key]
        logger.info(
            "_sdp_admm_blas: reusing cached Cholesky factor (cache_key=%s)", cache_key
        )
    else:
        logger.info(
            "_sdp_admm_blas: A_null=(%d×%d), computing G=A^T A via BLAS...",
            n_psd,
            n_null,
        )
        t0 = time.time()
        G = A_null.T @ A_null  # numpy DGEMM, (n_null × n_null)
        logger.info("_sdp_admm_blas: G computed in %.1f s", time.time() - t0)

        # Log G eigenvalue range to diagnose conditioning (cheap: O(n_null^2))
        g_eigs = np.linalg.eigvalsh(G)
        g_eig_min = float(g_eigs[0])
        g_eig_max = float(g_eigs[-1])
        g_cond = g_eig_max / max(abs(g_eig_min), 1e-15)
        logger.info(
            "_sdp_admm_blas: G eigenvalues: min=%.4e, max=%.4e, cond=%.4e "
            "(optimal rho ≈ %.4e)",
            g_eig_min,
            g_eig_max,
            g_cond,
            float(np.sqrt(abs(g_eig_min) * g_eig_max)),
        )

        sigma = 1e-8 * max(1.0, float(np.diag(G).mean()))
        M_factor = rho * G + sigma * np.eye(n_null, dtype=np.float64)
        t0 = time.time()
        F = scipy.linalg.cho_factor(M_factor, lower=True, check_finite=False)
        logger.info("_sdp_admm_blas: Cholesky done in %.1f s", time.time() - t0)

        if cache_key is not None:
            _ADMM_CACHE[cache_key] = F

    # --- ADMM loop ---
    # Constraint: z = A x + b_psd ∈ PSD, dual y for (z - s = 0)
    s = np.zeros(n_psd, dtype=np.float64)
    y = np.zeros(n_psd, dtype=np.float64)
    x_null = np.zeros(n_null, dtype=np.float64)

    # Precompute A^T b_psd once
    At_bpsd = A_null.T @ b_psd  # (n_null,)

    obj_best = np.inf
    x_best = None
    status = "max_iter_reached"
    s_prev = np.zeros(n_psd, dtype=np.float64)

    for k in range(maxiters):
        # x-update: (rho G) x = rho A^T (s - b_psd) - A^T y - c_null
        rhs = rho * (A_null.T @ s - At_bpsd) - A_null.T @ y - c_null
        x_null = scipy.linalg.cho_solve(F, rhs, check_finite=False)

        # z = A x + b_psd  (bootstrap matrix entries in SCS-vectorized form)
        z = A_null @ x_null + b_psd

        # Over-relaxation: z_hat = alpha*z + (1-alpha)*s  (alpha=1.5 same as SCS)
        # Damps oscillation that occurs with alpha=1 (no relaxation).
        z_hat = alpha * z + (1.0 - alpha) * s

        # s-update: s = proj_K(z_hat + y/rho)
        s_prev[:] = s
        s = _admm_proj_psd_blocks(z_hat + y / rho, psd_dims)

        # y-update: y += rho * (z_hat - s)
        residual = z_hat - s
        y += rho * residual

        # Logging + convergence check every log_interval steps
        if k % log_interval == 0:
            # Primal residual = ||z - s|| (actual primal infeasibility, not z_hat)
            r_primal = float(np.linalg.norm(z - s))
            r_dual = float(rho * np.linalg.norm(A_null.T @ (s - s_prev)))
            obj_k = float(c_null @ x_null)
            min_eig_k = _admm_min_psd_eig(z, psd_dims)
            logger.info(
                "_sdp_admm_blas k=%d: r_primal=%.2e r_dual=%.2e obj=%.6f min_eig=%.3e",
                k,
                r_primal,
                r_dual,
                obj_k,
                min_eig_k,
            )
            if min_eig_k >= -eps_abs and obj_k < obj_best:
                obj_best = obj_k
                x_best = x_null.copy()

            eps_p = eps_abs * np.sqrt(n_psd) + eps_rel * max(
                float(np.linalg.norm(z)), float(np.linalg.norm(s))
            )
            if r_primal < eps_p:
                logger.info(
                    "_sdp_admm_blas converged at k=%d (r_primal=%.2e < eps=%.2e)",
                    k,
                    r_primal,
                    eps_p,
                )
                status = "solved"
                break

    # Final diagnostics
    z_final = A_null @ x_null + b_psd
    min_eig_final = _admm_min_psd_eig(z_final, psd_dims)
    logger.info(
        "_sdp_admm_blas done: k=%d, status=%s, obj=%.6f, min_eig=%.4e",
        k,
        status,
        float(c_null @ x_null),
        min_eig_final,
    )
    if x_best is None:
        x_best = x_null  # return last iterate even if not certified feasible
    return x_best, status


def _sdp_scs_direct(
    linear_objective_vector: np.ndarray,
    bootstrap_table_sparse: csr_matrix,
    linear_inhomogeneous_eq: tuple,
    extra_bootstrap_tables: Optional[dict] = None,
    maxiters: int = 100000,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    eps_infeas: float = 1e-7,
    verbose: bool = False,
    use_indirect: bool = True,
) -> tuple[np.ndarray, str]:
    """Solve the bootstrap SDP directly via SCS, bypassing CVXPY.

    Builds the SCS A matrix from the bootstrap tables in O(nnz) time
    (no CVXPY Python-level coefficient extraction), then calls scs.solve().

    Returns (x_optimal, status_string) where x_optimal is the full-dim
    parameter vector satisfying the equality constraints.
    """
    import scs

    A_eq, b_eq = linear_inhomogeneous_eq
    n_params = bootstrap_table_sparse.shape[1]

    parts_A: list[csr_matrix] = [A_eq.tocsr()]
    parts_b: list[np.ndarray] = [b_eq]
    psd_sizes: list[int] = []

    # --- main bootstrap block ---
    d = int(np.sqrt(bootstrap_table_sparse.shape[0]))
    bt_real = bootstrap_table_sparse.real.astype(np.float64)
    bt_imag = bootstrap_table_sparse.imag.astype(np.float64)
    has_imag = bt_imag.nnz > 0 and np.max(np.abs(bt_imag.data)) > 1e-10
    if has_imag:
        psd_rows = _scs_complex_psd_rows(bt_real, bt_imag, d)
        psd_sizes.append(2 * d)
    else:
        psd_rows = _scs_real_psd_rows(bt_real, d)
        psd_sizes.append(d)
    parts_A.append((-psd_rows).tocsr())
    parts_b.append(np.zeros(psd_rows.shape[0]))

    # --- extra charge-sector blocks ---
    if extra_bootstrap_tables:
        for q in sorted(extra_bootstrap_tables.keys()):
            tq = extra_bootstrap_tables[q]
            d_q = int(np.sqrt(tq.shape[0]))
            tq_real = tq.real.astype(np.float64)
            tq_imag = tq.imag.astype(np.float64)
            has_imag_q = tq_imag.nnz > 0 and np.max(np.abs(tq_imag.data)) > 1e-10
            if has_imag_q:
                psd_q = _scs_complex_psd_rows(tq_real, tq_imag, d_q)
                psd_sizes.append(2 * d_q)
            else:
                psd_q = _scs_real_psd_rows(tq_real, d_q)
                psd_sizes.append(d_q)
            parts_A.append((-psd_q).tocsr())
            parts_b.append(np.zeros(psd_q.shape[0]))

    A_scs = sparse.vstack(parts_A, format="csc")
    b_scs = np.concatenate(parts_b)
    c_scs = linear_objective_vector.astype(np.float64).copy()

    cone = {"z": int(A_eq.shape[0]), "s": psd_sizes}
    data = {"A": A_scs, "b": b_scs, "c": c_scs}

    logger.info(
        "_sdp_scs_direct: A=(%d×%d, nnz=%d), PSD blocks=%s",
        A_scs.shape[0],
        A_scs.shape[1],
        A_scs.nnz,
        psd_sizes,
    )
    t0 = time.time()
    sol = scs.solve(
        data,
        cone,
        max_iters=maxiters,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        eps_infeas=eps_infeas,
        verbose=True,  # always log SCS progress
        use_indirect=use_indirect,
    )
    logger.info(
        "_sdp_scs_direct: status=%s, time=%.1f s",
        sol["info"]["status"],
        time.time() - t0,
    )
    return sol["x"], sol["info"]["status"]


def sdp_minimize_null(
    linear_objective_vector: np.ndarray,
    bootstrap_table_sparse: csr_matrix,
    linear_inhomogeneous_eq: tuple[csr_matrix, np.ndarray],
    null_space_projector,
    param_particular,
    radius: float = np.inf,
    maxiters: int = 2500,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    eps_infeas: float = 1e-7,
    reg: float = 1e-4,
    verbose: bool = False,
    cvxpy_solver: str = "SCS",
    use_factorization_block: bool = False,
    factorization_v_table: Optional[np.ndarray] = None,
    extra_bootstrap_tables: Optional[dict] = None,
    full_rank_linear_eq: Optional[tuple] = None,
    admm_rho: float = 1.0,
    admm_alpha: float = 1.5,
    admm_log_interval: int = 500,
    clarabel_static_reg: float = 1e-7,
) -> tuple[bool, str, np.ndarray]:
    """
    Performs the following SDP minimization over the vector variable x:

    A linear objective v^T x is minimized, subject to the constraints
        - The bootstrap matrix M(x) is positive semi-definite
        - The linear inhomogeneous equation A x = b is satisfied
        - A second linear inhomgeneous equation A' x = b' is imposed as a penalty
        in the objective function
        - The variable x is constrained to lie within a radius-R ball of an initial
        variable `init`

    Parameters
    ----------
    linear_objective_vector : np.ndarray
        A vector determining the objective function v^T x
    bootstrap_table_sparse : csr_matrix
        The bootstrap table (represented as a sparse CSR matrix)
    linear_inhomogeneous_eq : tuple[csr_matrix, np.ndarray]
        A tuple (A, b) of a matrix A and a vector b, which together
        define a linear inhomogeneous equation of the form A x = b.
        This equation will be imposed directly as a constraint.
    linear_inhomogeneous_penalty : tuple[csr_matrix, np.ndarra]
        A tuple (A, b) of a matrix A and a vector b, which together
        define a linear inhomogeneous equation of the form A x = b.
        This equation will be imposed indirectly as a penalty term
        added to the objective function.
    init : np.ndarray
        An initial value for the variable vector x.
    radius : float, optional
        The maximum allowable change in the initial vector, by default np.inf
    maxiters : int, optional
        The maximum number of iterations for the SDP optimization, by default 10_000
    eps : float, optional
        A tolerance variable for the SDP optimization, by default 1e-4
    reg : float, optional
        A regularization coefficient controlling the relative weight of the
        penalty term, by default 1e-4
    verbose : bool, optional
        An optional boolean flag used to set the verbosity, by default True

    Returns
    -------
    tuple[bool, str, np.ndarray]
        A tuple containing
            a boolean variable, where True corresponds to a successful execution of the optimization
            a str containing the optimization status
            a numpy array with the final, optimized vector
    """
    # CVXPY method (only SCS is supported for this particular convex problem)
    if cvxpy_solver == "SCS":
        solver = cp.SCS
    elif cvxpy_solver == "ECOS":
        solver = cp.ECOS
    elif cvxpy_solver == "OSQP":
        solver = cp.OSQP
    elif cvxpy_solver == "CLARABEL":
        solver = cp.CLARABEL
    elif cvxpy_solver == "MOSEK":
        solver = cp.MOSEK
    else:
        raise NotImplementedError

    # set-up
    matrix_dim = int(np.sqrt(bootstrap_table_sparse.shape[0]))

    # Parameterization strategy:
    #   SCS — full-dim (3998-var) + original sparse equality A_full (rank-deficient
    #     but sparse cyclic constraints).  use_indirect=True so CG is used for the
    #     ADMM x-update; CG doesn't require positive-definiteness of the equality
    #     block, only of (rho*I + A^T A) which is always PD.  SVD equality rows are
    #     DENSE (Vt rows), so they would make CG slow — avoid them for SCS.
    #
    #   CLARABEL with full_rank_linear_eq — full-dim (3998-var) + SVD-derived
    #     full-rank equality (1187×3998 dense).  Avoids KKT singularity for
    #     interior-point solvers, but CLARABEL still hits NumericalError near the
    #     PSD boundary.  Kept for experimentation.
    #
    #   MOSEK — full-dim + original equality (presolver handles rank deficit).
    #
    #   Others (ECOS, OSQP) — null-space (2811-dim); no equality constraints.
    _use_full_rank_eq = cvxpy_solver == "CLARABEL" and full_rank_linear_eq is not None

    # --- ADMM-BLAS path: null-space ADMM with BLAS Cholesky ---
    # Projects bootstrap tables into null-space coords via _build_null_psd_rows,
    # caches G = A_null^T A_null Cholesky, then runs _sdp_admm_blas.
    # Safe: no CVXPY overhead, no QDLDL, no A^T A fill-in blow-up.
    if cvxpy_solver == "SCS":
        ns = null_space_projector  # (n_params, n_null)
        pp = param_particular  # (n_params,)
        c_null = ns.T @ linear_objective_vector  # (n_null,)

        # Main bootstrap block
        d_main = int(np.sqrt(bootstrap_table_sparse.shape[0]))
        bt_real = bootstrap_table_sparse.real.astype(np.float64)
        bt_imag = bootstrap_table_sparse.imag.astype(np.float64)
        has_imag = bt_imag.nnz > 0 and np.max(np.abs(bt_imag.data)) > 1e-10
        A_null_main, b_psd_main = _build_null_psd_rows(
            bt_real, bt_imag, d_main, ns, pp, has_imag
        )
        psd_dims = [2 * d_main if has_imag else d_main]
        A_null_blocks = [A_null_main]
        b_psd_blocks = [b_psd_main]

        # Extra charge-sector blocks
        if extra_bootstrap_tables:
            for q in sorted(extra_bootstrap_tables.keys()):
                tq = extra_bootstrap_tables[q]
                d_q = int(np.sqrt(tq.shape[0]))
                tq_real = tq.real.astype(np.float64)
                tq_imag = tq.imag.astype(np.float64)
                has_imag_q = tq_imag.nnz > 0 and np.max(np.abs(tq_imag.data)) > 1e-10
                A_q, b_q = _build_null_psd_rows(
                    tq_real, tq_imag, d_q, ns, pp, has_imag_q
                )
                psd_dims.append(2 * d_q if has_imag_q else d_q)
                A_null_blocks.append(A_q)
                b_psd_blocks.append(b_q)

        A_null_all = np.vstack(A_null_blocks)
        b_psd_all = np.concatenate(b_psd_blocks)

        x_null, admm_status = _sdp_admm_blas(
            c_null=c_null,
            A_null=A_null_all,
            b_psd=b_psd_all,
            psd_dims=psd_dims,
            maxiters=maxiters,
            rho=admm_rho,
            alpha=admm_alpha,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            log_interval=admm_log_interval,
            cache_key=(id(ns), id(pp)),
        )

        # Reconstruct full-dim parameter vector
        x_full = ns @ x_null + pp

        # Diagnostics
        A_lin, b_lin = linear_inhomogeneous_eq
        M_val = bootstrap_table_sparse @ x_full
        M_r = M_val.real.reshape(d_main, d_main)
        M_i = M_val.imag.reshape(d_main, d_main)
        if has_imag and np.max(np.abs(M_i)) > 1e-10:
            M_bmat = np.block([[M_r, -M_i], [M_i, M_r]])
        else:
            M_bmat = M_r
        try:
            min_eig = float(np.linalg.eigvalsh(M_bmat)[0])
        except np.linalg.LinAlgError:
            min_eig = float("nan")

        return x_full, {
            "prob.status": admm_status,
            "prob.value": float(linear_objective_vector @ x_full),
            "maxiters_cvxpy": maxiters,
            "||x||": float(np.linalg.norm(x_full)),
            "violation_of_linear_constraints": float(
                np.linalg.norm(A_lin @ x_full - b_lin)
            ),
            "min_bootstrap_eigenvalue": min_eig,
        }

    if cvxpy_solver in ("MOSEK",) or _use_full_rank_eq:
        num_variables = bootstrap_table_sparse.shape[1]
        param_null = cp.Variable(num_variables)
        _use_null_space_param = False
    else:
        # ECOS, OSQP, CLARABEL (without full_rank_linear_eq): null-space
        num_variables = null_space_projector.shape[1]
        param_null = cp.Variable(num_variables)
        _use_null_space_param = True

    # Build _table_expr: maps a bootstrap/constraint table → CVXPY affine expression.
    if _use_null_space_param:
        _ns = null_space_projector
        _pp = param_particular

        def _table_expr(tbl):
            proj = tbl.dot(_ns)
            const = tbl.dot(_pp)
            return proj @ param_null + const

    else:

        def _table_expr(tbl):
            return tbl @ param_null

    # build the constraints
    # 1. the PSD bootstrap constraint(s)
    # 2. A @ param == 0
    # 3. ||param||_2 <= radius
    _t_build = time.time()

    # If the n x n bootstrap table is complex, split the corresponding bootstrap matrix
    # into a (2n, 2n) real matrix, rather than a (n, n) complex matrix.
    # Note: based on the param vector, it could be the case that a complex-valued bootstrap
    # table gives rise to a real bootstrap matrix - e.g., if odd-degree expectation values
    # vanish for the given param vector.
    # cvxpy's C++ backend requires contiguous float arrays; complex sparse matrices
    # produce non-contiguous views via .real/.imag. Cast explicitly to float64.
    if np.max(np.abs(bootstrap_table_sparse.imag)) > 1e-10:
        logger.debug("Mapping complex bootstrap matrix to real")
        bootstrap_table_sparse_real = bootstrap_table_sparse.real.astype(np.float64)
        bootstrap_table_sparse_imag = bootstrap_table_sparse.imag.astype(np.float64)
        matrix_real = cp.reshape(
            _table_expr(bootstrap_table_sparse_real),
            (matrix_dim, matrix_dim),
            order="F",
        )
        matrix_imag = cp.reshape(
            _table_expr(bootstrap_table_sparse_imag),
            (matrix_dim, matrix_dim),
            order="F",
        )
        matrix_block = cp.bmat(
            [[matrix_real, -matrix_imag], [matrix_imag, matrix_real]]
        )
        constraints = [matrix_block >> 0]
    else:
        constraints = [
            cp.reshape(
                _table_expr(bootstrap_table_sparse.real.astype(np.float64)),
                (matrix_dim, matrix_dim),
                order="F",
            )
            >> 0
        ]

    # Constrain the variable to a ball.
    # For null-space parameterization, null_space_projector is orthonormal so
    # ||N x|| = ||x||; bounding ||x||_null ≤ radius also bounds the null component.
    constraints += [cp.norm(param_null) <= radius]

    # Enforce equality constraints explicitly for full-dim solvers.
    if not _use_null_space_param:
        # CLARABEL with SVD full-rank equality: dense Vt rows (1187×3998).
        # SCS/MOSEK: original sparse A_full (rank-deficient OK for CG/presolver).
        if _use_full_rank_eq and full_rank_linear_eq is not None:
            constraints += [
                full_rank_linear_eq[0] @ param_null == full_rank_linear_eq[1]
            ]
        else:
            constraints += [
                linear_inhomogeneous_eq[0] @ param_null == linear_inhomogeneous_eq[1]
            ]

    # optional factorization-block constraint: M_hat = [[M, v], [v^T, 1]] ⪰ 0
    # This is a necessary condition for large-N factorization M_IJ = <O_I><O_J>
    # and bounds the SDP from below without making the problem non-convex.
    if use_factorization_block:
        if np.max(np.abs(bootstrap_table_sparse.imag)) > 1e-10:
            raise NotImplementedError(
                "factorization block not yet implemented for complex bootstrap tables"
            )
        if factorization_v_table is None:
            raise ValueError(
                "factorization_v_table must be provided when use_factorization_block=True"
            )
        n = matrix_dim
        v_table = factorization_v_table.astype(np.float64)
        M_cvxpy = cp.reshape(
            _table_expr(bootstrap_table_sparse.real.astype(np.float64)),
            (n, n),
            order="F",
        )
        v_expr = cp.reshape(_table_expr(v_table), (n, 1))
        M_hat = cp.bmat(
            [
                [M_cvxpy, v_expr],
                [v_expr.T, np.ones((1, 1))],
            ]
        )
        constraints.append(M_hat >> 0)

    # Extra PSD blocks for non-zero charge sectors (invariant basis only).
    # Each charge-q block has entries that are charge-0 expectation values, so
    # it constrains the same free parameters but provides additional PSD bounds.
    if extra_bootstrap_tables:
        for q, table_q in extra_bootstrap_tables.items():
            n_q = int(np.sqrt(table_q.shape[0]))
            if np.max(np.abs(table_q.imag)) > 1e-10:
                table_q_real = table_q.real.astype(np.float64)
                table_q_imag = table_q.imag.astype(np.float64)
                M_q_real = cp.reshape(_table_expr(table_q_real), (n_q, n_q), order="F")
                M_q_imag = cp.reshape(_table_expr(table_q_imag), (n_q, n_q), order="F")
                M_q_block = cp.bmat([[M_q_real, -M_q_imag], [M_q_imag, M_q_real]])
                constraints.append(M_q_block >> 0)
            else:
                M_q = cp.reshape(
                    _table_expr(table_q.real.astype(np.float64)), (n_q, n_q), order="F"
                )
                constraints.append(M_q >> 0)

    # the loss to minimize
    if linear_objective_vector is not None:
        if _use_null_space_param:
            # Project objective onto null space: obj^T (N x + p) = (N^T obj)^T x + const.
            # The constant doesn't affect the minimization.
            _obj_ns = null_space_projector.T.dot(linear_objective_vector)
            loss = _obj_ns @ param_null
        else:
            loss = linear_objective_vector @ param_null

        # add l2 regularization
        loss += reg * cp.norm(param_null)

    else:
        logger.debug("sdp_minimize_null: minimizing l2 norm of param")
        loss = cp.norm(param_null)

    # solve the optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints)
    logger.info(
        "CVXPY problem built in %.1f s (%s, %d vars, %d constraints)",
        time.time() - _t_build,
        cvxpy_solver,
        num_variables,
        len(constraints),
    )
    _t_solve = time.time()
    if cvxpy_solver == "SCS":
        # Full-dim (3998-var) + full-rank SVD equality (1187×3998 sparse) +
        # sparse bootstrap tables.  CVXPY canonicalization is fast (~6s).
        # INDIRECT (CG) mode avoids QDLDL factorization of A^T A, which fills
        # in to near-dense for this problem (28376×3998 with ~500K nnz).
        # Each ADMM step uses CG mat-vecs against sparse A; globally convergent.
        prob.solve(
            verbose=verbose,
            max_iters=maxiters,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_infeas=eps_infeas,
            solver=solver,
            use_indirect=True,  # INDIRECT (CG): avoids QDLDL fill-in on semi-dense A^T A
        )
    elif cvxpy_solver == "CLARABEL":
        # Full-rank SVD equality + full-dim (3998-var) + sparse bootstrap table:
        #   - No KKT singularity (equality block is now full row-rank 1187×3998).
        #   - Sparse Schur complement (B^T D B where B is sparse 5041×3998).
        #   - Expected ~10-15 s per iteration (vs 58 s with dense null-space projection).
        #   - Should converge to optimality in 30-50 iterations without NumericalError.
        prob.solve(
            verbose=verbose,
            max_iter=maxiters,
            tol_gap_abs=max(eps_abs, 1e-5),
            tol_gap_rel=max(eps_rel, 1e-5),
            tol_feas=max(eps_abs, 1e-5),
            static_regularization_constant=clarabel_static_reg,
            equilibrate_enable=True,
            equilibrate_max_iter=20,
            solver=solver,
        )
    elif cvxpy_solver == "MOSEK":
        prob.solve(
            verbose=verbose,
            # max_iters=maxiters,
            # eps_abs=eps_abs,
            # eps_rel=eps_rel,
            # eps_infeas=eps_infeas,
            solver=solver,
            accept_unknown=True,  # https://www.cvxpy.org/tutorial/solvers/index.html
            mosek_params={
                # "MSK_DPAR_OPTIMIZER_MAX_TIME": 100,
                # "MSK_IPAR_BI_MAX_ITERATIONS": 10_000_000,
                # "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000_00,
                # "MSK_IPAR_SIM_MAX_ITERATIONS": 30_000_000,
                #'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-9,
                #'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-9,
                #'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-9,
                #'MSK_IPAR_PRESOLVE_USE': 2,  # Full presolve
                #'MSK_IPAR_INTPNT_MAX_ITERATIONS': 1000,
            },
        )

    logger.info(
        "SDP solve done in %.1f s, status=%s", time.time() - _t_solve, prob.status
    )
    if param_null.value is None:
        raise ValueError("sdp_minimize failed, None value returned.")

    # Reconstruct the full-dimensional parameter vector.
    if _use_null_space_param:
        full_param_val = null_space_projector @ param_null.value + param_particular
    else:
        full_param_val = param_null.value

    # log information on the extent to which the constraints are satisfied
    ball_constraint = np.linalg.norm(full_param_val)
    violation_of_linear_constraints = np.linalg.norm(
        linear_inhomogeneous_eq[0] @ full_param_val - linear_inhomogeneous_eq[1]
    )
    min_bootstrap_eigenvalue = np.linalg.eigvalsh(
        (bootstrap_table_sparse @ full_param_val).reshape(matrix_dim, matrix_dim)
    )[0]
    logger.debug("sdp_minimize status: %s", prob.status)
    logger.debug("sdp_minimize ||A x - b||: %.4e", violation_of_linear_constraints)
    logger.debug(
        "sdp_minimize bootstrap matrix min eigenvalue: %.4e", min_bootstrap_eigenvalue
    )

    optimization_result = {
        "solver": cvxpy_solver,
        "prob.status": prob.status,
        "prob.value": prob.value,
        "maxiters_cvxpy": maxiters,
        "||x||": ball_constraint,
        "violation_of_linear_constraints": violation_of_linear_constraints,
        "min_bootstrap_eigenvalue": min_bootstrap_eigenvalue,
    }

    return full_param_val, optimization_result


def solve_bootstrap(
    bootstrap: BootstrapSystem,
    st_operator_to_minimize: SingleTraceOperator,
    st_operator_inhomo_constraints=[(SingleTraceOperator(data={(): 1}), 1)],
    init: Optional[np.ndarray] = None,
    maxiters: int = 25,
    maxiters_cvxpy: int = 2500,
    tol: float = 1e-5,
    init_scale: float = 1.0,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    eps_infeas: float = 1e-7,
    reg: float = 1e-4,
    PRNG_seed=None,
    radius: float = 1e8,
    cvxpy_solver: str = "SCS",
    use_factorization_block: bool = False,
    admm_rho: float = 1.0,
    admm_alpha: float = 1.5,
    admm_log_interval: int = 500,
    clarabel_static_reg: float = 1e-7,
) -> np.ndarray:
    """
    Solve the bootstrap by minimizing the objective function subject to
    the bootstrap constraints.

    TODO: add more info on the Newton's method used here.

    Parameters
    ----------
    bootstrap : BootstrapSystem
        The bootstrap system to be solved
    st_operator_to_minimize : SingleTraceOperator
        The single-trace operator whose expectation value we wish to minimize
    st_operator_inhomo_constraints : list, optional
        The single-trace expectation value constraints, <tr(O)>=c,
        by default [ (SingleTraceOperator(data={(): 1}), 1) ]
    init : Optional[np.ndarray], optional
        The initial parameter vector, by default None
    maxiters : int, optional
        The maximum number of iterations for the optimization, by default 25
    tol : float, optional
        The tolerance for the quadratic constraint violations, by default 1e-8
    init_scale : float, optional
        An overall scale for the parameter vector, by default 1.0
    eps : float, optional
        The epsilon used in the cvxpy inner optimization problem, by default 1e-4
    reg : float, optional
        The regularization parameter for the penalty terms, by default 1e-4

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if PRNG_seed is not None:
        np.random.seed(PRNG_seed)
        logger.debug("Setting PRNG seed to %s", PRNG_seed)

    # get the bootstrap constraints necessary for the optimization
    # linear constraints
    if bootstrap.linear_constraints is None:
        _ = bootstrap.build_linear_constraints().tocsr()

    # quadratic constraints
    if bootstrap.quadratic_constraints_numerical is None:
        bootstrap.build_quadratic_constraints()
    quadratic_constraints_numerical = bootstrap.quadratic_constraints_numerical

    # bootstrap table
    if bootstrap.bootstrap_table_sparse is None:
        bootstrap.build_bootstrap_table()
    bootstrap_table_sparse = bootstrap.bootstrap_table_sparse

    # factorization-block v-table (only built when needed)
    factorization_v_table = (
        bootstrap.build_augmented_bootstrap_table() if use_factorization_block else None
    )

    # confirm bootstrap table is consistent with hermitian bootstrap matrix
    logger.debug("Bootstrap table dtype: %s", bootstrap_table_sparse.dtype)
    bootstrap_matrix_tmp = bootstrap_table_sparse @ np.random.normal(
        size=bootstrap.param_dim_null
    )
    bootstrap_matrix_tmp = bootstrap_matrix_tmp.reshape(
        (bootstrap.bootstrap_matrix_dim, bootstrap.bootstrap_matrix_dim)
    )
    if not ishermitian(bootstrap_matrix_tmp, atol=1e-12):
        raise ValueError("Error, bootstrap matrix is not Hermitian.")
    logger.debug("Bootstrap parameter dimension: %d", bootstrap.param_dim_null)

    # initialize the variable vector
    if init is None:
        logger.debug("Initializing randomly with init_scale=%.2e", init_scale)
        init = init_scale * np.random.normal(size=bootstrap.param_dim_null)
    else:
        init = np.asarray(init)
        logger.debug("Initializing from provided param vector")
    param_array = init
    _param_array_old = None  # reserved for future convergence check

    # map the single trace operator whose expectation value we wish to minimize to a coefficient vector
    if st_operator_to_minimize is not None:
        linear_objective_vector = bootstrap.single_trace_to_coefficient_vector(
            st_operator_to_minimize, return_null_basis=True
        )
        if not np.allclose(
            linear_objective_vector.imag, np.zeros_like(linear_objective_vector)
        ):
            raise ValueError(
                "Error, the coefficient vector is complex but should be real."
            )
        linear_objective_vector = linear_objective_vector.real
    else:
        linear_objective_vector = None

    # build the A, b matrix and vector for the linear inhomogeneous constraints
    # this will always include the constraint that tr(1) = 1, and possibly other constraints as well
    A = sparse.csr_matrix((0, bootstrap.param_dim_null))
    b = np.zeros(0)
    for op, value in st_operator_inhomo_constraints:

        A = sparse.vstack(
            [
                A,
                sparse.csr_matrix(
                    bootstrap.single_trace_to_coefficient_vector(
                        op, return_null_basis=True
                    )
                ),
            ]
        )
        b = np.append(b, value)
    linear_inhomogeneous_eq_no_quadratic = (A, b)

    # When all "quadratic" constraints are actually linear (quadratic_term == 0),
    # the Newton iteration is unnecessary: all constraints can be imposed at step 0.
    # Pre-compute the null space of the full constraint system (original + cyclic
    # linear constraints) ONCE here to avoid a ~30-min dense SVD inside the loop.
    _null_space_cache = None  # (null_space_matrix, param_particular)
    _full_rank_eq_cache = None  # (A_eq_full_rank sparse, b_eq_full_rank) from SVD
    _quadratic_is_linear_only = quadratic_constraints_numerical["quadratic"].nnz == 0
    if (
        _quadratic_is_linear_only
        and quadratic_constraints_numerical["linear"].shape[0] > 0
    ):
        A_linear = quadratic_constraints_numerical["linear"]
        A_full = sparse.vstack(
            [linear_inhomogeneous_eq_no_quadratic[0], A_linear]
        ).tocsr()
        b_full = np.append(
            linear_inhomogeneous_eq_no_quadratic[1],
            np.zeros(A_linear.shape[0]),
        )
        A_full_dense = np.asarray(A_full.todense())
        logger.info(
            "All quadratic constraints are linear: pre-computing null space of "
            "(%d x %d) constraint matrix (once, before Newton loop).",
            A_full_dense.shape[0],
            A_full_dense.shape[1],
        )
        # Compute thin SVD of A_full ONCE — reuse for consistency check, particular
        # solution, null space, and full-rank equality constraints (avoids 3-4 separate
        # SVD calls that each take ~1 min for this 3180×3998 matrix).
        logger.info(
            "Computing thin SVD of (%d x %d) constraint matrix (once)...",
            A_full_dense.shape[0],
            A_full_dense.shape[1],
        )
        U_svd, s_svd, Vt_svd = np.linalg.svd(A_full_dense, full_matrices=False)
        rank_tol = max(s_svd[0] * 1e-8, 1e-10)
        rank_svd = int(np.sum(s_svd > rank_tol))
        logger.info("SVD rank = %d (tol=%.2e)", rank_svd, rank_tol)

        # Null space: right singular vectors for zero singular values.
        null_space_full = Vt_svd[rank_svd:, :].T  # shape: (n_params, n_params - rank)

        # Compute particular solution (minimum-norm least-squares) via pseudoinverse.
        # pp = Vt[:rank].T @ diag(1/s[:rank]) @ U[:,:rank].T @ b
        def _svd_lstsq(b_vec):
            """Min-norm least-squares solution using precomputed SVD."""
            return Vt_svd[:rank_svd, :].T @ (
                (U_svd[:, :rank_svd].T @ b_vec) / s_svd[:rank_svd]
            )

        # Check if b_full is in the row space of A_full.  If not, the system is
        # inconsistent — this happens when simplify_quadratic=True collapsed the
        # normalization row into A_linear with a wrong (zero) RHS.
        b_proj = U_svd[:, :rank_svd] @ (U_svd[:, :rank_svd].T @ b_full)
        _resid_test = float(np.linalg.norm(b_full - b_proj))
        if _resid_test > 1e-4:
            # The normalization direction A_norm is in the row space of A_linear
            # (simplify_quadratic=True collapsed a cyclic constraint row into A_norm,
            # setting its RHS to 0 instead of the correct nonzero value).
            # Fix: correct the RHS for all A_linear rows:
            #   alpha_k = A_linear[k] @ A_norm_vec / ||A_norm_vec||^2
            # so the corrected system [A_norm; A_linear] @ x = [1; alpha] IS consistent.
            logger.warning(
                "Quadratic constraints inconsistent with normalization "
                "(||b - proj(b)||=%.4f > 1e-4). Correcting RHS via projection of "
                "A_norm onto A_linear rows to obtain consistent 2811-dim null space.",
                _resid_test,
            )
            _A_norm_vec = (
                np.asarray(linear_inhomogeneous_eq_no_quadratic[0].todense())
                .ravel()
                .astype(np.float64)
            )
            _norm_sq = float(np.dot(_A_norm_vec, _A_norm_vec))
            # A_linear rows are rows 1: of A_full_dense
            _alpha = A_full_dense[1:, :] @ _A_norm_vec / _norm_sq  # (n_linear,)
            _b_corrected = np.concatenate([[1.0], _alpha])
            _pp_corrected = _svd_lstsq(_b_corrected)
            _resid_corrected = float(
                np.linalg.norm(A_full_dense @ _pp_corrected - _b_corrected)
            )
            _norm_check = float(_A_norm_vec @ _pp_corrected)
            logger.info(
                "Corrected RHS: ||A pp - b_corrected||=%.2e, A_norm @ pp=%.6f "
                "(nonzero alpha entries: %d/%d)",
                _resid_corrected,
                _norm_check,
                int(np.sum(np.abs(_alpha) > 1e-10)),
                len(_alpha),
            )
            if _resid_corrected > 1e-6:
                logger.error(
                    "Corrected RHS still inconsistent (||resid||=%.2e). "
                    "Falling back to normalization-only null space.",
                    _resid_corrected,
                )
                _A_norm_dense = np.asarray(
                    linear_inhomogeneous_eq_no_quadratic[0].todense()
                )
                param_particular_full = np.linalg.lstsq(
                    _A_norm_dense,
                    linear_inhomogeneous_eq_no_quadratic[1],
                    rcond=None,
                )[0]
                null_space_full = get_null_space_dense(matrix=_A_norm_dense)
            else:
                param_particular_full = _pp_corrected
                # null_space_full already computed from A_full_dense SVD above
                # Update b_full so full-rank equality cache uses consistent RHS.
                b_full = _b_corrected
        else:
            param_particular_full = _svd_lstsq(b_full)
        logger.info("Pre-computed null space: %d dimensions.", null_space_full.shape[1])
        _null_space_cache = (null_space_full, param_particular_full)

        # Full-rank equality constraints from the same SVD.
        # A_full (3180×3998) has rank 1187 with 1993 redundant rows.
        # Passing the rank-deficient A_full directly to CLARABEL causes a
        # singular KKT system.  Instead, use the SVD row-space basis:
        #   A_full = U Σ Vt  =>  equivalent full-rank system: Vt_r x = b_eq
        # where Vt_r (1187×3998) has orthonormal rows and b_eq = (U_r^T b) / σ.
        A_eq_full_rank = Vt_svd[:rank_svd, :]  # (rank × n_params)
        b_eq_full_rank = (U_svd[:, :rank_svd].T @ b_full) / s_svd[:rank_svd]  # (rank,)
        logger.info(
            "Full-rank equality: %d independent rows (SVD rank), tol=%.2e",
            rank_svd,
            rank_tol,
        )
        _full_rank_eq_cache = (sparse.csr_matrix(A_eq_full_rank), b_eq_full_rank)

        # Override the no-quadratic system so step 0 already uses all constraints.
        linear_inhomogeneous_eq_no_quadratic = (A_full, b_full)

    # iterate over steps
    for step in range(maxiters):

        # build the Newton method update for the quadratic constraints, which
        # produces a second inhomogenous linear equation A' x = b'
        # here, the new equation is grad * (new_param - param) + val = 0
        # this equation will be imposed as a penalty
        quad_cons_val, quad_cons_grad = get_quadratic_constraint_vector(
            quadratic_constraints_numerical, param_array, compute_grad=True
        )
        logger.debug(
            "step %d/%d: objective=%.4f, max_quad_violation=%.4e, ||param||=%.4e, radius=%.4e, reg=%.4e",
            step + 1,
            maxiters,
            (
                linear_objective_vector @ param_array
                if linear_objective_vector is not None
                else float("nan")
            ),
            np.max(np.abs(quad_cons_val)) if quad_cons_val.size else 0.0,
            np.linalg.norm(param_array),
            radius,
            reg,
        )

        # build the Ax=b constraints
        # includes <1>=1, optional constraints like <O>=e, and the linearized
        # quadratic constraints
        if step == 0:
            linear_inhomogeneous_eq = linear_inhomogeneous_eq_no_quadratic
        else:
            linear_inhomogeneous_eq = (
                sparse.vstack(
                    [linear_inhomogeneous_eq_no_quadratic[0], quad_cons_grad]
                ),
                np.append(
                    linear_inhomogeneous_eq_no_quadratic[1],
                    np.asarray(quad_cons_grad.dot(param_array) - quad_cons_val)[0],
                ),
            )

        # build a penalty term to enforce Ax=b for the linearized quadratic constraints
        # NOTE only used for the original, non-null sdp_minimize method
        # linear_inhomogeneous_penalty = (
        #    quad_cons_grad,
        #    np.asarray(quad_cons_grad.dot(param_array) - quad_cons_val)[0],
        # )

        # get the null space projector
        A, b = linear_inhomogeneous_eq
        A = A.todense()

        if _null_space_cache is not None:
            # Use the pre-computed null space (linear-only case or cached from a
            # prior step).  This avoids a repeated 30-min dense SVD at every step.
            null_space_matrix, param_particular = _null_space_cache
        else:
            param_particular = np.linalg.lstsq(A, b, rcond=None)[0]
            null_space_matrix = get_null_space_dense(matrix=A)
        # Pass the thin null-space matrix directly so the CVXPY variable has
        # only the free (non-degenerate) coordinates. The square projection
        # null_space_matrix @ pinv(null_space_matrix) is rank-deficient and
        # causes NumericalError in interior-point solvers (e.g. CLARABEL).
        null_space_projector = null_space_matrix

        # perform the inner convex minimization
        try:
            param_array_soln, optimization_result = sdp_minimize_null(
                linear_objective_vector=linear_objective_vector,
                bootstrap_table_sparse=bootstrap_table_sparse,
                linear_inhomogeneous_eq=linear_inhomogeneous_eq,
                null_space_projector=null_space_projector,
                param_particular=param_particular,
                radius=radius,
                maxiters=maxiters_cvxpy,
                eps_abs=eps_abs,
                eps_rel=eps_rel,
                eps_infeas=eps_infeas,
                reg=reg,
                verbose=False,
                cvxpy_solver=cvxpy_solver,
                use_factorization_block=use_factorization_block,
                factorization_v_table=factorization_v_table,
                extra_bootstrap_tables=getattr(
                    bootstrap, "extra_bootstrap_tables", None
                ),
                full_rank_linear_eq=_full_rank_eq_cache,
                admm_rho=admm_rho,
                admm_alpha=admm_alpha,
                admm_log_interval=admm_log_interval,
                clarabel_static_reg=clarabel_static_reg,
            )

        except Exception as e:
            logger.warning("sdp_minimize_null failed at step %d: %s", step + 1, e)
            return None, None

        # interpolate between the old and new solutions
        alpha = 1.0
        if step < 2:
            param_array = param_array_soln
        else:
            param_array = alpha * param_array_soln + (1 - alpha) * param_array

        # print out some diagnostic information
        quad_cons_val = get_quadratic_constraint_vector(
            quadratic_constraints_numerical, param_array, compute_grad=False
        )
        max_quad_constraint_violation = (
            np.max(np.abs(quad_cons_val)) if quad_cons_val.size else 0.0
        )
        quad_constraint_violation_norm = np.linalg.norm(quad_cons_val)
        optimization_result["max_quad_constraint_violation"] = (
            max_quad_constraint_violation
        )
        optimization_result["quad_constraint_violation_norm"] = (
            quad_constraint_violation_norm
        )

        logger.info(
            "step %d/%d: objective=%.4f, max_quad_violation=%.4e, ||param||=%.4e",
            step + 1,
            maxiters,
            (
                linear_objective_vector @ param_array
                if linear_objective_vector is not None
                else float("nan")
            ),
            max_quad_constraint_violation,
            np.linalg.norm(param_array),
        )

        # add the seed to the result
        optimization_result["PRNG_seed"] = PRNG_seed

        # record the boostrap matrix
        optimization_result["bootstrap_matrix_real"] = bootstrap.get_bootstrap_matrix(
            param=param_array
        ).real.tolist()
        optimization_result["bootstrap_matrix_imag"] = bootstrap.get_bootstrap_matrix(
            param=param_array
        ).imag.tolist()

        # For the linear-only case (all "quadratic" constraints are linear),
        # the null space and all constraints are fixed at step 0.  Every
        # subsequent step solves the identical SDP, so one step is enough.
        if _null_space_cache is not None:
            logger.info(
                "Linear-only constraints: exiting Newton loop after step 1 "
                "(no benefit from further iterations)."
            )
            return param_array, optimization_result

        # terminate early if the tolerance is satisfied
        if step > (10 - 2) and max_quad_constraint_violation < tol:
            return param_array, optimization_result

    return param_array, optimization_result
