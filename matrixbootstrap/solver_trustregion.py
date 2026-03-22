from typing import Union

import cvxpy as cp
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import (
    csr_matrix,
    vstack,
)

import logging

from matrixbootstrap.algebra import SingleTraceOperator
from matrixbootstrap.bootstrap import BootstrapSystem
from matrixbootstrap.linear_algebra import get_null_space_dense

logger = logging.getLogger(__name__)


def get_null_space_quantities(
    bootstrap: BootstrapSystem,
    st_operator_inhomo_constraints: list[tuple[SingleTraceOperator, float]],
    quadratic_constraints_numerical,
    param_array: np.ndarray,
    include_quadratic:bool=True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    bootstrap : BootstrapSystem
        _description_
    st_operator_inhomo_constraints : list[tuple[SingleTraceOperator, float]]
        _description_
    quadratic_constraints_numerical : _type_
        _description_
    param_array : np.ndarray
        _description_
    include_quadratic : bool, optional
        _description_, by default True

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        _description_
    """

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

    # combine the constraints from op_cons and linearized quadratic constraints, i.e., grad * (new_param - param) + val = 0
    val, grad = get_quadratic_constraint_vector_sparse(
        quadratic_constraints_numerical, param_array, compute_grad=True
    )

    # build the Ax=b constraints
    # includes <1>=1, optional constraints like <O>=e, and the linearized
    # quadratic constraints
    if not include_quadratic:
        linear_inhomogeneous_eq = linear_inhomogeneous_eq_no_quadratic
    else:
        linear_inhomogeneous_eq = (
            sparse.vstack([linear_inhomogeneous_eq_no_quadratic[0], grad]),
            np.append(
                linear_inhomogeneous_eq_no_quadratic[1],
                np.asarray(grad.dot(param_array) - val)[0],
            ),
        )

    # get the null space projector
    A, b = linear_inhomogeneous_eq
    A = A.todense()
    param_particular = np.linalg.lstsq(A, b, rcond=None)[0]
    null_space_matrix = get_null_space_dense(matrix=A)
    null_space_projector = null_space_matrix @ np.linalg.pinv(null_space_matrix)

    return null_space_projector, param_particular, A, b


def sdp_init(
    bootstrap_table_sparse: sparse.csr_matrix,
    null_space_projector: np.ndarray,
    param_particular: np.ndarray,
    init_array: np.ndarray,
    maxiters:int=5_000,
    verbose:bool=False,
    cvxpy_solver:str="SCS",
    eps_abs:float=1e-5,
    eps_rel:float=1e-5,
    eps_infeas:float=1e-7,
):
    """
    Finds the parameters such that
            1. All bootstrap tables are positive semidefinite;
            2. ||param|| is minimized.
    Arguments:
            tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then
                    has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square
                    matrices. The constraint is that these matrices must be positive semidefinite.
            A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)):
                    linear constraints that A.dot(param) = b
            init (numpy array of shape (num_variables,)): the initial parameters
            reg (float): regularization
            maxiters (int), eps (float), verbose (bool): options for the convex solver
    Returns:
            param (numpy array of shape (num_variables,)): the optimal parameters found
    """

    # write param vector as particular solution plus a null term
    num_variables = init_array.size
    param_null = cp.Variable(num_variables)  # declare the cvxpy param
    param = null_space_projector @ param_null + param_particular

    # the PSD constraint(s) (multiple if bootstrap matrix is block diagonal)
    size = int(np.sqrt(bootstrap_table_sparse.shape[0]))
    constraints = [cp.reshape(bootstrap_table_sparse @ param, (size, size)) >> 0]

    # solve the above described optimization problem
    prob = cp.Problem(cp.Minimize(cp.norm(param)), constraints)

    if cvxpy_solver == "SCS":
        prob.solve(
            verbose=verbose,
            max_iters=maxiters,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_infeas=eps_infeas,
            solver=cvxpy_solver,
        )
    elif cvxpy_solver == "MOSEK":
        prob.solve(
            verbose=verbose,
            solver=cvxpy_solver,
            accept_unknown=True,  # https://www.cvxpy.org/tutorial/solvers/index.html
        )

    if str(prob.status) != "optimal":
        logger.warning("sdp_init unexpected status: %s", prob.status)

    return param.value


def sdp_relax(
    bootstrap_table_sparse: sparse.csr_matrix,
    null_space_projector: np.ndarray,
    param_particular: np.ndarray,
    init_array: np.ndarray,
    radius: float,
    maxiters:int=10_000,
    relax_rate:float=0.8,
    verbose:bool=False,
    cvxpy_solver:str="SCS",
    eps_abs:float=1e-5,
    eps_rel:float=1e-5,
    eps_infeas:float=1e-7,
):
    """
    Finds the parameters such that
            1. All bootstrap tables are positive semidefinite;
            2. ||param - init||_2 <= 0.8 * radius;
            3. The violation of linear constraints ||A.dot(param) - b||_2 is minimized.
    Arguments:
            tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then
                    has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square
                    matrices. The constraint is that these matrices must be positive semidefinite.
            A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)):
                    linear constraints that A.dot(param) = b
            init (numpy array of shape (num_variables,)): the initial parameters
            radius (float): radius of the trust region
            maxiters (int), eps (float), verbose (bool): options for the convex solver
    Returns:
            param (numpy array of shape (num_variables,)): the optimal parameters found
    """

    # write param vector as particular solution plus a null term
    num_variables = init_array.size
    param_null = cp.Variable(num_variables)  # declare the cvxpy param
    param = null_space_projector @ param_null + param_particular

    # build the constraints
    # 1. ||param - init||_2 <= relax_rate * radius
    # 2. the PSD bootstrap constraint(s)
    constraints = [cp.norm(param - init_array) <= relax_rate * radius]
    size = int(np.sqrt(bootstrap_table_sparse.shape[0]))
    constraints = [cp.reshape(bootstrap_table_sparse @ param, (size, size)) >> 0]

    # solve the above described optimization problem
    prob = cp.Problem(cp.Minimize(cp.norm(param)), constraints)

    if cvxpy_solver == "SCS":
        prob.solve(
            verbose=verbose,
            max_iters=maxiters,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_infeas=eps_infeas,
            solver=cvxpy_solver,
        )
    elif cvxpy_solver == "MOSEK":
        prob.solve(
            verbose=verbose,
            solver=cvxpy_solver,
            accept_unknown=True,  # https://www.cvxpy.org/tutorial/solvers/index.html
        )

    if str(prob.status) != "optimal":
        logger.warning("sdp_relax unexpected status: %s", prob.status)

    return param.value


def sdp_minimize(
    vec,
    bootstrap_table_sparse,
    null_space_projector,
    param_particular,
    A,
    b,
    init_array,
    radius,
    reg=1e-4,
    maxiters=10_000,
    verbose:bool=False,
    cvxpy_solver:str="SCS",
    eps_abs:float=1e-5,
    eps_rel:float=1e-5,
    eps_infeas:float=1e-7,
):
    """
    Finds the parameters such that
            1. All bootstrap tables are positive semidefinite;
            2. ||param - init||_2 <= radius;
            3. A.dot(param) = b;
            4. vec.dot(param) + reg * np.linalg.norm(param) is minimized.
    Arguments:
            vec (numpy array of shape (num_variables,))
            tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then
                    has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square
                    matrices. The constraint is that these matrices must be positive semidefinite.
            A (sparse matrix of shape (num_constraints, num_variables)), b (numpy array of shape (num_constraints,)):
                    linear constraints that A.dot(param) = b
            init (numpy array of shape (num_variables,)): the initial parameters
            radius (float): radius of the trust region
            reg (float): regularization parameter
            maxiters (int), eps (float), verbose (bool): options for the convex solver
    Returns:
            param (numpy array of shape (num_variables,)): the optimal parameters found
    """
    # write param vector as particular solution plus a null term
    num_variables = init_array.size
    param_null = cp.Variable(num_variables)  # declare the cvxpy param
    param = null_space_projector @ param_null + param_particular

    # build the constraints
    # 1. ||param - init||_2 <= radius
    # 2. the PSD bootstrap constraint(s)
    # 3. A @ param == 0
    constraints = [cp.norm(param - init_array) <= radius]
    size = int(np.sqrt(bootstrap_table_sparse.shape[0]))
    constraints = [cp.reshape(bootstrap_table_sparse @ param, (size, size)) >> 0]

    # the loss to minimize
    loss = vec @ param + reg * cp.norm(param)

    # solve the above described optimization problem
    prob = cp.Problem(cp.Minimize(loss), constraints)

    if cvxpy_solver == "SCS":
        prob.solve(
            verbose=verbose,
            max_iters=maxiters,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_infeas=eps_infeas,
            solver=cvxpy_solver,
        )
    elif cvxpy_solver == "MOSEK":
        prob.solve(
            verbose=verbose,
            solver=cvxpy_solver,
            accept_unknown=True,  # https://www.cvxpy.org/tutorial/solvers/index.html
        )

    if str(prob.status) != "optimal":
        logger.warning("sdp_minimize unexpected status: %s", prob.status)

    if param.value is None:
        return None, None

    # log information on the extent to which the constraints are satisfied
    ball_constraint = np.linalg.norm(param.value - init_array)
    violation_of_linear_constraints = np.linalg.norm(A @ param.value - b)
    min_bootstrap_eigenvalue = np.linalg.eigvalsh(
        (bootstrap_table_sparse @ param.value).reshape(size, size)
    )[0]
    logger.debug("sdp_minimize status (maxiters=%d): %s", maxiters, prob.status)
    logger.debug("sdp_minimize ||A x - b||: %.4e", violation_of_linear_constraints)
    logger.debug("sdp_minimize bootstrap matrix min eigenvalue: %.4e", min_bootstrap_eigenvalue)

    optimization_result = {
        "solver": cvxpy_solver,
        "prob.status": prob.status,
        "prob.value": prob.value,
        "maxiters_cvxpy": maxiters,
        "||x-init||": ball_constraint,
        "violation_of_linear_constraints": violation_of_linear_constraints,
        "min_bootstrap_eigenvalue": min_bootstrap_eigenvalue,
    }

    return param.value, optimization_result


def get_quadratic_constraint_vector_dense(
    quadratic_constraints: dict[str, np.ndarray],
    param: np.ndarray,
    compute_grad: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Compute the quadratic constraint vector
        A_{Iij} K_{ia} K_{jb} u_a u_b + B_{Ii} K_{Ia} u_a
    and optionally the gradient
        A_{Iij} K_{ia} K_{jb} u_b + A_{Iij} K_{ia} K_{jb} u_a + B_{Ii} K_{Ia}
    for a given parameter vector (in the null space) u.

    Parameters
    ----------
    quadratic_constraints : dict[str, np.ndarray]
        The quadratic and linear parts of the quadratic/cyclic constraints,
        represented as arrays.
    param : np.ndarray
        The parameter vector in the null space, u.
    compute_grad : bool, optional
        Controls whether the grad is computed, by default False.

    Returns
    -------
    Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
        The constraint vector and optionally its gradient.
    """

    # compute the constraints
    # linear term
    linear_term = quadratic_constraints["linear"] @ param

    # quadratic term
    param_product = np.outer(param, param).reshape((len(param) ** 2))
    quadratic_array = quadratic_constraints["quadratic"]
    quadratic_term = quadratic_constraints["quadratic"] @ param_product

    constraint_vector = linear_term + quadratic_term

    # return the constraint only if the gradient is not needed
    if not compute_grad:
        return constraint_vector

    # compute the gradient
    num_constraints = linear_term.shape[0]
    quadratic_array = np.asarray(quadratic_constraints["quadratic"].todense())
    quadratic_array = quadratic_array.reshape((num_constraints, len(param), len(param)))
    # compute the gradient matrix
    constraint_grad_quadratic_term_1 = np.einsum("Iij, i -> Ij", quadratic_array, param)
    constraint_grad_quadratic_term_2 = np.einsum("Iij, j -> Ii", quadratic_array, param)
    constraint_grad = (
        quadratic_constraints["linear"]
        + constraint_grad_quadratic_term_1
        + constraint_grad_quadratic_term_2
    )

    return constraint_vector, constraint_grad


def get_quadratic_constraint_vector_sparse(
    quadratic_constraints: dict[str, np.ndarray],
    param: np.ndarray,
    compute_grad: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Compute the quadratic constraint vector
        A_{Iij} K_{ia} K_{jb} u_a u_b + B_{Ii} K_{Ia} u_a
    and optionally the gradient
        A_{Iij} K_{ia} K_{jb} u_b + A_{Iij} K_{ia} K_{jb} u_a + B_{Ii} K_{Ia}
    for a given parameter vector (in the null space) u.

    Parameters
    ----------
    quadratic_constraints : dict[str, np.ndarray]
        The quadratic and linear parts of the quadratic/cyclic constraints,
        represented as arrays.
    param : np.ndarray
        The parameter vector in the null space, u.
    compute_grad : bool, optional
        Controls whether the grad is computed, by default False.

    Returns
    -------
    Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
        The constraint vector and optionally its gradient.
    """

    # compute the constraints
    # linear term
    linear_term = quadratic_constraints["linear"] @ param

    # quadratic term
    param_product = np.outer(param, param).reshape((len(param) ** 2))
    quadratic_term = quadratic_constraints["quadratic"] @ param_product

    constraint_vector = linear_term + quadratic_term

    # return the constraint only if the gradient is not needed
    if not compute_grad:
        return constraint_vector

    # compute the gradient
    num_constraints = linear_term.shape[0]
    quad_terms = [
        quadratic_constraints["quadratic"][i].reshape((len(param), len(param)))
        for i in range(num_constraints)
    ]
    quad_terms = vstack(
        [
            csr_matrix(quad_term @ param + quad_term.T @ param)
            for quad_term in quad_terms
        ]
    )
    constraint_grad = quadratic_constraints["linear"] + quad_terms

    # I found that this is critical, without it the result is wrong
    constraint_grad = constraint_grad.todense()

    return constraint_vector, constraint_grad


def solve_bootstrap(
    bootstrap: BootstrapSystem,
    st_operator_to_minimize: SingleTraceOperator,
    st_operator_inhomo_constraints=[(SingleTraceOperator(data={(): 1}), 1)],
    maxiters: int = 25,
    maxiters_cvxpy: int = 2500,
    tol: float = 1e-5,
    init_scale: float = 1e-2,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    eps_infeas: float = 1e-7,
    reg: float = 1e-4,
    PRNG_seed=None,
    radius: float = 1e8,
    cvxpy_solver: str = "SCS",
    eps=1e-4,
    verbose=True,
):
    """
    Minimizes the operator subject to the bootstrap positivity constraint, the quadratic cyclicity constraint, and the operator values
    constraints. The algorithm is a trust-region sequential semidefinite programming with regularization on l2 norm of the parameters.
    Arguments:
            op (TraceOperator): expectation value of this operator is to be minimized
            tables (list of tuples (int, sparse matrix)): the first integer gives the size of the bootstrap matrix and the sparse matrix then
                    has shape (size * size, num_variables), dot the matrix with the parameters and reshape into a list of real symmetric square
                    matrices. The constraint is that these matrices must be positive semidefinite.
            quad_cons (QuadraticSolution): the quadratic constraints from solving the cyclicity constraints
            op_cons (list of tuples (TraceOperator, float)): the extra constraints specifying the expectation values of the given operators
            init (numpy array of shape (num_variables,)): the initial value of the parameters
            eps (float): the accuracy goal
            reg (float): regulariation on parameters. The true objective to minimize will be expectation value of op + reg * l2-norm(param)
            verbose (bool): whether to print the optimization progress
            savefile (string): the filename to save the parameters
    Returns:
            param (numpy array of shape (num_variables,)): the optimal value of parameters found
    """
    relax_rate = 0.8

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

    logger.debug("Bootstrap parameter dimension: %d", bootstrap.param_dim_null)

    # initialize the variable vector
    init_array = init_scale * np.random.normal(size=bootstrap.param_dim_null)

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

    # the loss function to minimize, i.e., the value of op
    # vec = operator_to_vector(sol, op)
    loss = lambda param: linear_objective_vector.dot(param)

    null_space_projector, param_particular, A, b = get_null_space_quantities(
        bootstrap,
        st_operator_inhomo_constraints,
        quadratic_constraints_numerical,
        init_array,
        include_quadratic=False,
    )

    # find an initial parameter close to init that makes all bootstrap matrices positive
    param = sdp_init(
        bootstrap_table_sparse=bootstrap_table_sparse,
        null_space_projector=null_space_projector,
        param_particular=param_particular,
        init_array=init_array,
        verbose=False,
        cvxpy_solver=cvxpy_solver,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        eps_infeas=eps_infeas,
    )
    radius = np.linalg.norm(param) + 20

    # penalty parameter for violation of constraints
    mu = 1

    # optimization steps
    for step in range(maxiters):
        logger.debug("step %d/%d (radius=%.4e, mu=%.4e)", step + 1, maxiters, radius, mu)

        # one step
        null_space_projector, param_particular, A, b = get_null_space_quantities(
            bootstrap,
            st_operator_inhomo_constraints,
            quadratic_constraints_numerical,
            param,
            include_quadratic=True,
        )

        '''
        relaxed_param = sdp_relax(
            bootstrap_table_sparse=bootstrap_table_sparse,
            null_space_projector=null_space_projector,
            param_particular=param_particular,
            init=param,
            radius=radius,
            relax_rate=relax_rate,
            verbose=False,
            maxiters=maxiters_cvxpy,
            cvxpy_solver=cvxpy_solver,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_infeas=eps_infeas,
        )
        '''

        null_space_projector, param_particular, A, b = get_null_space_quantities(
            bootstrap,
            st_operator_inhomo_constraints,
            quadratic_constraints_numerical,
            param,
            include_quadratic=True,
        )

        param_new, optimization_result = sdp_minimize(
            vec=linear_objective_vector,
            bootstrap_table_sparse=bootstrap_table_sparse,
            null_space_projector=null_space_projector,
            param_particular=param_particular,
            A=A,
            b=b,
            init_array=param,
            radius=radius,
            reg=reg,
            verbose=False,
            maxiters=maxiters_cvxpy,
            cvxpy_solver=cvxpy_solver,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            eps_infeas=eps_infeas,
        )

        if param_new is None:
            # wrongly infeasible
            radius *= relax_rate  # GSH used to be 0.9
            logger.warning("Wrongly infeasible at step %d, expanding radius to %.4e", step + 1, radius)
            continue

        # compute constraint violations for candidate new parameters
        linear_constraint_violation = A.dot(param_new) - b
        max_linear_constraint_violation = np.max(np.abs(linear_constraint_violation))

        quadratic_constraint_violation = get_quadratic_constraint_vector_sparse(
            quadratic_constraints_numerical, param_new
        )
        max_quadratic_constraint_violation = np.max(np.abs(quadratic_constraint_violation))

        min_bootstrap_eigenvalue = np.linalg.eigvalsh(
            (bootstrap_table_sparse @ param_new).reshape(
                bootstrap.bootstrap_matrix_dim, bootstrap.bootstrap_matrix_dim
            )
        )[0]

        logger.info(
            "step %d/%d: loss=%.4f, max_lin_viol=%.4e, max_quad_viol=%.4e, min_eig=%.4e, R=%.4e",
            step + 1, maxiters, loss(param_new),
            max_linear_constraint_violation, max_quadratic_constraint_violation,
            min_bootstrap_eigenvalue, radius,
        )

        if (
            step > 4
            and max_linear_constraint_violation < eps
            and max_quadratic_constraint_violation < eps
            and min_bootstrap_eigenvalue > -eps
            and np.linalg.norm(param_new - param) < eps * radius
        ):
            logger.info("Accuracy goal achieved at step %d.", step + 1)
            return param_new, optimization_result

        # compute the changes in the merit function and decide whether the step should be accepted
        dloss = (
            loss(param_new)
            - loss(param)
            + reg * (np.linalg.norm(param_new) - np.linalg.norm(param))
        )

        array1 = np.squeeze(np.asarray(A.dot(param_new) - b))
        array2 = get_quadratic_constraint_vector_sparse(
            quadratic_constraints_numerical, param_new
        )  # 1D array

        dcons = np.linalg.norm(np.append(array1, array2, axis=0)) - np.linalg.norm(
            linear_constraint_violation
        )

        if -dcons > eps:
            old_mu = mu
            mu = max(mu, -2 * dloss / dcons)
            logger.debug("  adjusting mu from %.4e to %.4e", old_mu, mu)

        # the merit function is -loss(param) - reg * norm(param) - mu * norm(constraint vector)
        acred = -dcons  # actual constraint reduction
        ared = -dloss + mu * acred  # actual merit increase
        pcred = np.linalg.norm(linear_constraint_violation) - np.linalg.norm(
            A.dot(param_new) - b
        )  # predicted constraint reduction
        pred = -dloss + mu * pcred  # predicted merit increase
        rho = ared / pred  # some sanity checks

        if rho > 0.5:
            # accept
            if max_linear_constraint_violation < eps and min_bootstrap_eigenvalue > -eps:
                radius *= 2 - relax_rate  # GSH used to be 1.2
                logger.debug("  step accepted, expanding R to %.4e", radius)
            param = param_new

        else:
            # reject
            old_radius = radius
            radius = relax_rate * np.linalg.norm(param_new - param)
            logger.debug("  step rejected, shrinking R from %.4e to %.4e", old_radius, radius)

    logger.warning("minimize did not converge to precision %.5f within %d steps.", eps, maxiters)
    return param, optimization_result
