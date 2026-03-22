import logging
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.nn import ReLU
from torch.optim.lr_scheduler import ExponentialLR

from matrixbootstrap.algebra import SingleTraceOperator
from matrixbootstrap.bootstrap import BootstrapSystem
from matrixbootstrap.linear_algebra import get_null_space_dense

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
torch_dtype = torch.float64


def solve_bootstrap(
    bootstrap: BootstrapSystem,
    st_operator_to_minimize: SingleTraceOperator,
    st_operator_inhomo_constraints=[(SingleTraceOperator(data={(): 1}), 1)],
    init: Optional[np.ndarray] = None,
    PRNG_seed=None,
    init_scale: float = 1e-2,
    lr=1e-1,
    gamma=0.999,
    num_epochs=10_000,
    penalty_reg=1e2,
    tol=1e-6,
    patience=100,
    early_stopping_tol=1e-3,
) -> np.ndarray:

    logger.info(f"torch device: {device}")

    if PRNG_seed is not None:
        np.random.seed(PRNG_seed)
        torch.manual_seed(PRNG_seed)
        logger.debug(f"setting PRNG seed to {PRNG_seed}")

    # get the bootstrap constraints necessary for the optimization
    # linear constraints
    if bootstrap.linear_constraints is None:
        _ = bootstrap.build_linear_constraints().tocsr()

    # quadratic constraints
    if bootstrap.quadratic_constraints_numerical is None:
        bootstrap.build_quadratic_constraints()

    # bootstrap table
    if bootstrap.bootstrap_table_sparse is None:
        bootstrap.build_bootstrap_table()
    logger.debug("Bootstrap parameter dimension: %d", bootstrap.param_dim_null)

    # build the Ax = b constraints
    A, b = [], []
    for st_operator, val in st_operator_inhomo_constraints:
        A.append(
            bootstrap.single_trace_to_coefficient_vector(
                st_operator, return_null_basis=True
            )
        )
        b.append(val)
    A = np.asarray(A)  # convert to numpy array
    b = np.asarray(b)

    # get the null space projector
    A_null_space = get_null_space_dense(matrix=A)
    null_space_projector = A_null_space @ np.linalg.pinv(A_null_space)

    # convert to torch tensor
    A = torch.from_numpy(A).type(torch_dtype).to(device)
    b = torch.from_numpy(b).type(torch_dtype).to(device)
    null_space_projector = (
        torch.from_numpy(null_space_projector).type(torch_dtype).to(device)
    )

    # get the vector of the operator to bound (minimize)
    vec = bootstrap.single_trace_to_coefficient_vector(
        st_operator_to_minimize, return_null_basis=True
    )
    vec = torch.from_numpy(vec).type(torch_dtype).to(device)

    # build the bootstrap array
    bootstrap_table = torch.from_numpy(bootstrap.bootstrap_table_sparse.todense()).to(
        device
    )

    # build the constraints
    quadratic_constraints = bootstrap.quadratic_constraints_numerical
    quadratic_constraint_linear = (
        torch.from_numpy(quadratic_constraints["linear"].todense())
        .type(torch_dtype)
        .to(device)
    )
    quadratic_constraint_quadratic = (
        torch.from_numpy(quadratic_constraints["quadratic"].todense())
        .type(torch_dtype)
        .to(device)
    )
    quadratic_constraint_quadratic = quadratic_constraint_quadratic.reshape(
        (
            len(quadratic_constraint_quadratic),
            bootstrap.param_dim_null,
            bootstrap.param_dim_null,
        )
    )

    def get_full_param(param_null, param_particular):
        return null_space_projector @ param_null + param_particular

    def operator_loss(param_null, param_particular):
        param = get_full_param(param_null, param_particular)
        expectation_value = vec @ param
        return expectation_value

    def get_quadratic_constraint_vector(param):
        quadratic_constraints = torch.einsum(
            "Iab, a, b -> I", quadratic_constraint_quadratic, param, param
        ) + torch.einsum("Ia, a -> I", quadratic_constraint_linear, param)
        return torch.square(quadratic_constraints)

    def quadratic_loss(param_null, param_particular):
        param = get_full_param(param_null, param_particular)
        return torch.norm(get_quadratic_constraint_vector(param))

    def quadratic_constraint_max(param_null, param_particular):
        param = get_full_param(param_null, param_particular)
        return torch.max(torch.abs(get_quadratic_constraint_vector(param)))

    def Axb_loss(param_null, param_particular):
        param = get_full_param(param_null, param_particular)
        return torch.norm(A @ param - b)

    def psd_loss(param_null, param_particular):
        param = get_full_param(param_null, param_particular)

        # complex bootstrap matrix
        if torch.max(torch.abs(bootstrap_table.imag)) > 1e-10:
            # debug("Mapping complex bootstrap table to real") # this prints every single iteration, not just evey 100
            bootstrap_table_real = bootstrap_table.real
            bootstrap_table_imag = bootstrap_table.imag
            matrix_real = (bootstrap_table_real @ param).reshape(
                (bootstrap.bootstrap_matrix_dim, bootstrap.bootstrap_matrix_dim)
            )
            matrix_imag = (bootstrap_table_imag @ param).reshape(
                (bootstrap.bootstrap_matrix_dim, bootstrap.bootstrap_matrix_dim)
            )
            # stack them
            top_row = torch.cat((matrix_real, -matrix_imag), dim=1)
            bottom_row = torch.cat((matrix_imag, matrix_real), dim=1)
            bootstrap_matrix = torch.cat((top_row, bottom_row), dim=0)

        # real bootstrap matrix
        else:
            bootstrap_matrix = (bootstrap_table.real @ param).reshape(
                (bootstrap.bootstrap_matrix_dim, bootstrap.bootstrap_matrix_dim)
            )
        # smallest_eigv = torch.linalg.eigvalsh(bootstrap_matrix)[0]
        # smallest_eigv = torch.real(smallest_eigv)
        # return ReLU()(-smallest_eigv)

        # return the l2 norm of the vector of negative eigenvalues
        return torch.sqrt(
            torch.sum(
                torch.square(
                    ReLU()(-torch.real(torch.linalg.eigvalsh(bootstrap_matrix)))
                )
            )
        )

    def build_loss(param_null, param_particular, penalty_reg=penalty_reg):
        loss = (
            operator_loss(param_null, param_particular)
            + penalty_reg * psd_loss(param_null, param_particular)
            + penalty_reg * quadratic_loss(param_null, param_particular)
            + penalty_reg * Axb_loss(param_null, param_particular)
            # + 1e2 * torch.abs(torch.linalg.norm(get_full_param(param_null, param_particular)) - 1e3)
        )
        return loss

    # initialize the variable vector
    if init is None:
        init = init_scale * np.random.randn(bootstrap.param_dim_null)
        param_particular = np.linalg.lstsq(
            A.cpu().numpy(), b.cpu().numpy(), rcond=None
        )[0]
        param_null = init
        logger.debug(
            f"Initializing param to be the least squares solution of Ax=b plus Gaussian noise with scale = {init_scale}."
        )
    else:
        raise ValueError
        param_null = init
        logger.debug(f"Initializing as param={init}")

    param_null = torch.tensor(param_null).type(torch_dtype).to(device)
    param_null = null_space_projector @ param_null

    param_null.requires_grad = True
    param_particular = torch.tensor(param_particular).type(torch_dtype).to(device)

    # optimizer
    optimizer = optim.Adam([param_null], lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        loss = build_loss(param_null=param_null, param_particular=param_particular)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():

            if ((epoch + 1) % 100 == 0) or (epoch == num_epochs - 1):
                total_loss = loss.detach().cpu().item()
                operator_value = (
                    operator_loss(
                        param_null=param_null, param_particular=param_particular
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                violation_of_linear_constraints = (
                    Axb_loss(param_null=param_null, param_particular=param_particular)
                    .detach()
                    .cpu()
                    .item()
                )
                min_bootstrap_eigenvalue = (
                    psd_loss(param_null=param_null, param_particular=param_particular)
                    .detach()
                    .cpu()
                    .item()
                )
                quad_constraint_violation_norm = (
                    quadratic_loss(
                        param_null=param_null, param_particular=param_particular
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                _quad_constraint_violation_max = (
                    quadratic_constraint_max(
                        param_null=param_null, param_particular=param_particular
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                param_norm = torch.linalg.norm(
                    get_full_param(param_null, param_particular)
                )

                logger.debug(
                    f"epoch: {epoch+1}/{num_epochs}, lr: {scheduler.get_last_lr()[0]:.3e} total_loss: {total_loss:.3e}: op_loss: {operator_value:.5f}, ||x||: {param_norm:.3e}, ||Ax-b||: {violation_of_linear_constraints:.3e}, min_eig: {min_bootstrap_eigenvalue:.3e}, ||quad cons||: {quad_constraint_violation_norm:.3e}"
                )

        # clean up
        del loss

    param_final = get_full_param(param_null, param_particular)

    optimization_result = {
        "solver": "pytorch",
        "num_epochs": num_epochs,
        "operator_loss": float(
            operator_loss(param_null=param_null, param_particular=param_particular)
            .detach()
            .cpu()
            .item()
        ),
        "violation_of_linear_constraints": float(
            Axb_loss(param_null=param_null, param_particular=param_particular)
            .detach()
            .cpu()
            .item()
        ),
        "min_bootstrap_eigenvalue": float(
            psd_loss(param_null=param_null, param_particular=param_particular)
            .detach()
            .cpu()
            .item()
        ),
        "quad_constraint_violation_norm": float(
            quadratic_loss(param_null=param_null, param_particular=param_particular)
            .detach()
            .cpu()
            .item()
        ),
        "max_quad_constraint_violation": float(
            torch.max(torch.abs(get_quadratic_constraint_vector(param_final)))
            .detach()
            .cpu()
            .item()
        ),
    }

    # convert to a list of floats (no numpy float types, so that the result can be saved as a json later)
    param_final = [float(x) for x in list(param_final.detach().cpu().numpy())]

    return param_final, optimization_result
