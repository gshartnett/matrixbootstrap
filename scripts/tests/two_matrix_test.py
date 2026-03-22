import logging

from matrixbootstrap.config_utils import (
    generate_config_two_matrix,
    run_all_configs,
)

logging.basicConfig(level=logging.INFO)

L = 3
g2 = 1.0
g4 = 1

energy = 1.5
st_operator_to_minimize = "x_2"

generate_config_two_matrix(
    config_filename="test",
    config_dir=f"TwoMatrix_L_{L}_test",
    checkpoint_path=f"TwoMatrix_L_{L}_symmetric_g2_{g2}_g4_{g4}",
    g2=g2,
    g4=g4,
    max_degree_L=L,
    impose_symmetries=True,
    load_from_previously_computed=True,
    odd_degree_vanish=True,
    simplify_quadratic=True,
    st_operator_to_minimize=st_operator_to_minimize,
    st_operators_evs_to_set={"energy": energy},
    # optimization_method="pytorch",
    init_scale=1e-1,
    optimization_method="newton",
    cvxpy_solver="SCS",
    reg=1e-4,
    eps_abs=1e-7,
    eps_rel=1e-7,
)

# execute
run_all_configs(
    config_dir=f"TwoMatrix_L_{L}_test", parallel=False, check_if_exists_already=False
)
