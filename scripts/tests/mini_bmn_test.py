import logging

from matrixbootstrap.config_utils import (
    generate_config_bmn,
    run_all_configs,
)

logging.basicConfig(level=logging.INFO)

# generate the config files
L = 3
lambd = 1
nu = 1.0
energy = 1.5
st_operator_to_minimize = "x_2"

generate_config_bmn(
    config_dir=f"MiniBMN_L_{L}_test",
    nu=nu,
    lambd=lambd,
    max_degree_L=L,
    impose_symmetries=True,
    load_from_previously_computed=True,
    odd_degree_vanish=False,
    simplify_quadratic=True,
    st_operator_to_minimize=st_operator_to_minimize,
    st_operators_evs_to_set={"energy": energy},
    # optimization_method="pytorch",
    optimization_method="newton",
    cvxpy_solver="SCS",
    # maxiters=1,
    # reg=1e-5,
)

# execute
run_all_configs(
    config_dir=f"MiniBMN_L_{L}_test", parallel=False, check_if_exists_already=False
)
