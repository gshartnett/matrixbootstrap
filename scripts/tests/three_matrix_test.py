import logging

import numpy as np

from matrixbootstrap.config_utils import (
    generate_config_three_matrix,
    run_all_configs,
)

logging.basicConfig(level=logging.INFO)

L = 3

lambd = 1
nu = 1

g2 = float(np.round(nu**2, decimals=6))
g4 = lambd
# g3 = float(3 * nu * np.sqrt(lambd)) # set g3 to zero, while keeping g2 and g4 related as in the MiniBFSS model
g3 = 0.1

energy = 2.5
st_operator_to_minimize = "x_2"

generate_config_three_matrix(
    config_filename="test",
    config_dir=f"ThreeMatrix_L_{L}_test",
    checkpoint_path=f"ThreeMatrix_L_{L}_symmetric_g2_{g2}_g3_{g3}_g4_{g4}",
    g2=g2,
    g3=g3,
    g4=g4,
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
    reg=1e-5,
)

# execute
run_all_configs(
    config_dir=f"ThreeMatrix_L_{L}_test", parallel=False, check_if_exists_already=False
)
