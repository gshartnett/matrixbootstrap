from matrixbootstrap.config_utils import (
    generate_configs_one_matrix,
    run_all_configs,
)

L = 3
dir = f"OneMatrix_L_{L}_play"

generate_configs_one_matrix(
    config_filename="test",
    config_dir=dir,
    checkpoint_dir=dir,
    g2=1,
    g4=1,
    g6=0,
    max_degree_L=3,
    maxiters_cvxpy=10_000_000,
    eps=1e-6,
    load_from_previously_computed=True,
)

# execute
run_all_configs(config_dir=dir, parallel=False)
